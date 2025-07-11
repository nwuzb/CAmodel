# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # 对于128x128正方形目标，采用智能填充策略而不是强制改变长宽比
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        current_aspect = float(bbox[2]) / bbox[3]
        
        # 如果目标是正方形（128x128），则保持原始长宽比并选择最大边进行正方形扩展
        if abs(target_aspect - 1.0) < 0.01:  # 目标是正方形
            # 使用最大边作为正方形边长
            max_size = max(bbox[2], bbox[3])
            
            # 计算中心点
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            
            # 创建正方形边界框
            bbox[0] = center_x - max_size / 2
            bbox[1] = center_y - max_size / 2
            bbox[2] = max_size
            bbox[3] = max_size
        else:
            # 对于非正方形目标，使用原始的长宽比调整
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int64)

    # clip at image boundaries - 修改逻辑允许部分超出边界的框
    # 只有当框完全在图像外部时才返回None
    img_height, img_width = image.shape[:2]
    
    # 检查是否完全在图像外部
    if (bbox[2] <= 0 or bbox[0] >= img_width or 
        bbox[3] <= 0 or bbox[1] >= img_height):
        return None
    
    # 对部分超出边界的框进行裁剪，但仍然提取特征
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray([img_width - 1, img_height - 1]), bbox[2:])
    
    # 确保裁剪后的框仍然有效（宽度和高度大于0）
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    
    # 使用智能缩放+填充，保持长宽比
    h, w = image.shape[:2]
    target_h, target_w = patch_shape
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    
    # 计算新的尺寸
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 缩放图像
    if new_w > 0 and new_h > 0:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, (1, 1), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(resized, (min(target_w//4, 8), min(target_h//4, 8)), 
                           interpolation=cv2.INTER_CUBIC)
        new_w, new_h = resized.shape[1], resized.shape[0]
    
    # 创建目标尺寸的黑色画布
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # 计算居中位置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # 将缩放后的图像放置在画布中心
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        self.session = tf.compat.v1.Session()
        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        tf.import_graph_def(graph_def, name="net")

        self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % input_name)
        self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
            "%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


class SavedModelEncoder(object):
    """TensorFlow SavedModel格式的特征提取器类"""
    
    def __init__(self, model_dir, input_name="images", output_name="features"):
        # 加载SavedModel格式模型
        self.model = tf.saved_model.load(model_dir)
        
        # 查找模型的输入输出签名
        self.signatures = list(self.model.signatures.keys())
        self.input_name = input_name
        self.output_name = output_name
        
        # 假设模型有默认签名
        self.infer = self.model.signatures["serving_default"]
        
        # 获取输入和输出张量名称
        self.input_tensor_name = list(self.infer.structured_input_signature[1].keys())[0]
        
        # 获取输出特征维度
        # 由于SavedModel的特性，我们可能需要实际运行一次模型来获取特征维度
        # 确保输入是float32类型，更新为128x128
        dummy_input = np.zeros((1, 128, 128, 3), dtype=np.float32)  # 更新为128x128
        dummy_output = self.infer(**{self.input_tensor_name: tf.convert_to_tensor(dummy_input)})
        
        # 获取输出特征维度和输入图像形状
        output_key = list(dummy_output.keys())[0]
        self.feature_dim = dummy_output[output_key].shape[-1]
        self.image_shape = [128, 128, 3]  # 更新为128x128图像形状
        
        # print(f"加载SavedModel成功。输入签名: {self.input_tensor_name}, 特征维度: {self.feature_dim}")
        
    def __call__(self, data_x, batch_size=32):
        # 数据预处理: 确保数据是浮点数且在[0,1]范围内
        data_x = data_x.astype(np.float32) / 255.0
        
        # 创建输出数组
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        
        # 分批处理数据
        for i in range(0, len(data_x), batch_size):
            batch = data_x[i:i+batch_size]
            # 运行推理
            batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)  # 明确指定为float32类型
            results = self.infer(**{self.input_tensor_name: batch_tensor})
            
            # 提取特征
            output_key = list(results.keys())[0]
            features = results[output_key].numpy()
            out[i:i+len(batch)] = features
            
        return out


# =========================  PyTorch Encoder  ==========================
# 允许直接加载 .pth 权重，省去 TF / PB 转换。

import torch
import torch.nn.functional as F
try:
    from torchvision import transforms as _tvf
except ImportError:
    _tvf = None  # 若缺失 torchvision，会在 __init__ 报错

try:
    from efficientnet_pytorch import EfficientNet as _EffNet
except ImportError:
    _EffNet = None


class TorchEncoder:
    """Load EfficientNet-B0 .pth and generate 128-dim embeddings (L2-norm)."""

    def __init__(self, pth_path: str, input_size: int = 128, feat_dim: int = 128, device: str = "mps"):
        if _EffNet is None or _tvf is None:
            raise ImportError("torchvision 或 efficientnet_pytorch 未安装，无法使用 TorchEncoder")

        self.device = torch.device(device)
        self.input_size = input_size

        # ---------- build network ----------
        backbone = _EffNet.from_name("efficientnet-b0")
        backbone._fc = torch.nn.Identity()
        in_feat = backbone._conv_head.out_channels

        class _ReIDModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.fc = torch.nn.Linear(in_feat, feat_dim)
                self.bn = torch.nn.BatchNorm1d(feat_dim)

            def forward(self, x):
                f = self.backbone(x)
                f = self.fc(f)
                f = self.bn(f)
                f = F.normalize(f, p=2, dim=1)
                return f

        self.model = _ReIDModel().to(self.device)
        self.model.load_state_dict(torch.load(pth_path, map_location=self.device), strict=False)
        self.model.eval()

        # ---------- preprocess ----------
        self.pre = _tvf.Compose([
            _tvf.ToPILImage(),
            _tvf.Resize((input_size, input_size)),
            _tvf.ToTensor(),
            _tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 供 create_box_encoder 使用
        self.image_shape = [input_size, input_size, 3]

    @torch.no_grad()
    def __call__(self, patches, batch_size: int = 32):
        # patches: uint8 BGR (H,W,C)
        tensors = []
        for img in patches:
            # BGR → RGB
            img_rgb = img[:, :, ::-1]
            tensors.append(self.pre(img_rgb))
        data = torch.stack(tensors).to(self.device)

        feats = []
        for i in range(0, len(data), batch_size):
            out = self.model(data[i:i + batch_size])
            feats.append(out.cpu())
        feats = torch.cat(feats, dim=0).numpy()
        return feats


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    """
    创建一个编码器函数，根据边界框提取特征
    
    参数:
    model_filename: 字符串，模型文件路径或SavedModel目录
    input_name: 模型输入张量名称
    output_name: 模型输出张量名称
    batch_size: 批处理大小
    
    返回:
    encoder: 函数，接受图像和边界框列表，返回对应的特征向量
    """
    # ---- 根据模型类型选择加载器 ----
    if model_filename.endswith(".pth"):
        # 直接使用 PyTorch + EfficientNet
        image_encoder = TorchEncoder(model_filename)

    elif model_filename.endswith(".onnx"):
        # 可选：在安装了 onnxruntime 时使用
        try:
            import onnxruntime as ort

            class OnnxEncoder:
                def __init__(self, onnx_path):
                    self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                    self.input_name = self.sess.get_inputs()[0].name
                    self.output_name = self.sess.get_outputs()[0].name
                    self.image_shape = [128, 128, 3]
                def __call__(self, data_x, batch_size=32):
                    data_x = data_x.astype("float32") / 255.0
                    return self.sess.run([self.output_name], {self.input_name: data_x})[0]

            image_encoder = OnnxEncoder(model_filename)
        except ImportError:
            raise ImportError("请先 pip install onnxruntime 或使用 .pth/.pb/.savedmodel")

    elif os.path.isdir(model_filename):
        # print(f"检测到SavedModel目录: {model_filename}")
        try:
            # 使用SavedModel加载器
            image_encoder = SavedModelEncoder(model_filename, input_name, output_name)
            # print(f"成功加载SavedModel: {model_filename}")
        except Exception as e:
            print(f"加载SavedModel失败: {e}")
            raise
    else:
        # 使用原始的冻结图模型加载器
        # print(f"加载冻结图模型文件: {model_filename}")
        try:
            image_encoder = ImageEncoder(model_filename, input_name, output_name)
            # print(f"成功加载冻结图模型: {model_filename}")
        except Exception as e:
            print(f"加载冻结图模型失败: {e}")
            raise
            
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int64)
        min_frame_idx = frame_indices.astype(np.int64).min()
        max_frame_idx = frame_indices.astype(np.int64).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
