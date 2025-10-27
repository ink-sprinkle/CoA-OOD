import torchvision.transforms as T

augment = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color jittering
    T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0),  # random block
])

def augment_image(img: np.ndarray) -> np.ndarray:
    """
    img: (C,H,W) uint8 或 float32 [0,1]
    return: (C,H,W) float32 [0,1]
    """
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # 转成 (H,W,C)
    img = np.transpose(img, (1, 2, 0))
    pil = Image.fromarray(img)
    aug = augment(pil)
    arr = np.array(aug).astype(np.float32) / 255.0
    # 转回 (C,H,W)
    return np.transpose(arr, (2, 0, 1))