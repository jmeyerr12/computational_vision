# texture segmentation assignment - Joao Meyer Muhlmann - GRR20232347
from pathlib import Path
import cv2
import numpy as np

OUT_DIR="output"
SIZE = 512
WINDOW=32
SCALES=[2,4,8]
ANGLES=[0,22.5,45,67.5,90,112.5,135]
N_GROUPS=4

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) # read img and make it grayscale
    if img is None:
        print("Not possible to open {path}")
        return
    if img.shape != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE))
        
    return img
    
def apply_filters(img):
    img = img.astype(np.float32) / 255.0
    answers = []
    
    for sigma in SCALES:
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k+=1
    
        # 7 gabor filters (texture in different directions)        
        for angle in ANGLES:
            theta = np.deg2rad()
            
            kernel = cv2.getGaborKernel(
                (k,k), # filter size
                sigma, # scale
                theta, # orientation,
                4*sigma, # wave length
                0.5, # filter format
                0, 
                ktype = cv2.CV_32F,
            )
            ans = cv2.filter2D(img, cv2.CV_32F, kernel)
            answers.append(np.abs(ans))
        
        blur = cv2.GaussianBlur(img, (k,k), sigma) # gaussian
        log = cv2.Laplacian(blur, cv2.CV_32F) # laplacian filter
        answers.append(np.abs(log))
    return answers
    
def extract_vectors(img):
    answers = apply_filters(img)
    vectors = []
    
    for y in range(0, SIZE, WINDOW):
        for x in range(0, SIZE, WINDOW):
            vector = []
            
            for ans in answers: # figure out
                region = ans[y: y + WINDOW, x: x + WINDOW]
                vector.append(np.mean(region))
                
            vectors.append(vector)
    return np.array(vectors, dtype=np.float32)
                

def main():
    OUT_DIR.mkdir(exist_ok=True)
    
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    archives = sorted(
        p for p in OUT_DIR.iterdir()
        if p.suffix.lower() in extensions
    )
    if not archives:
        print("Place images in directory 'images'")
        return
    if len(archives) < 32:
        print(f"Warning: less than 32 images")
        
    all_features = []
    images = []
    
    # feature extraction
    for path in archives:
        image = load_image(path)
        features = extract_vectors
        
        images.append((path, image, features))
        all_features.append(features)
        
    # join all img regions
    x = np.vstack(all_features)
    
    # normalize each of the 24 features (avoids one characteristic dominating distance for having larger numbers)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    x = (x - mean) / std
    x = x.astype(np.float32)
    
    # grouping (uses euclidian distance)
    
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.1,
    )
    
    cv2.setRNGSeed(42)

    _, labels, _ = cv2.kmeans(
        x,
        N_GROUPS,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()
    
    # segmented img creation

    
        

if __name__ == "__main__":
    main()