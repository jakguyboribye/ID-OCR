import cv2
import numpy as np
import argparse
import os

MIN_DIM = 1000

def reorient(img):
    _,w_og=img.shape

    template = cv2.imread('template/template.jpg')
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    if template.shape[1] >w_og//2:
        template = template[:,:w_og//2 -1]
    template_upside_down =cv2.rotate(template,cv2.ROTATE_180)

    #upright
    match = cv2.matchTemplate(image=img[:,:w_og//2], 
                            templ=template, 
                            method=cv2.TM_CCOEFF_NORMED)

    l1,_ = np.where(match >= 0.25)

    found1 = len(l1)>0
    match = cv2.matchTemplate(image=img[:,w_og//2:], 
                            templ=template_upside_down, 
                            method=cv2.TM_CCOEFF_NORMED)

    l2,_ = np.where(match >= 0.25)
    found2 = len(l2)>0
    if found1== found2:
        return img
    
    if found2:
        return cv2.rotate(img,cv2.ROTATE_180)
    return img

def enhance_gray_sharp(image):
    """Convert to grayscale, upscale if needed, apply unsharp-mask sharpening."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale =1.0
    if h>w:
        gray=cv2.rotate(gray,cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = gray.shape

    if h < MIN_DIM:
        scale = MIN_DIM / h
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray=reorient(gray)
    gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0*pow(scale,2))
    sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    return sharp

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Enhance cropped card images (grayscale + sharpen) for OCR"
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Manual source directory to image",
    )
    

    args = parser.parse_args()
    if args.source:
        filename=args.source
        extensions = (".png", ".jpg", ".jpeg")
        if filename.lower().endswith(extensions):
            img = cv2.imread(filename)
            og_name = os.path.basename(filename).split('/')[-1]
            if og_name.endswith("_crop"):
                og_name=og_name[:-5]
            new_name = og_name+"_gray_sharp.jpg"
            cv2.imwrite(new_name,enhance_gray_sharp(img))
        else:
            raise ValueError("Invalid File. Accept \".png\", \".jpg\", \".jpeg\"")
    else:
        raise ValueError("Source file is not given")


if __name__ == "__main__":
    main()