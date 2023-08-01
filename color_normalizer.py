import numpy as np
import cv2
"""
Ref: http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
"""
class MacenkoColorNormalization:
    def __init__(self, Io=240, alpha=1, beta=0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
        
    def __call__(self, img):
        img = np.array(img) # Convert PIL Image to numpy array
        h, w, c = img.shape
        img = img.reshape((-1,3))
        
        #h, w, c = 224, 224, 3
        #img = img.reshape((-1,3))
        OD = -np.log10((img.astype(np.float64)+1)/self.Io)
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        try:
            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        except np.linalg.LinAlgError:
            print("Error")
            return img
        else:
            That = ODhat.dot(eigvecs[:,1:3])
            phi = np.arctan2(That[:,1],That[:,0])
            minPhi = np.percentile(phi, self.alpha)
            maxPhi = np.percentile(phi, 100-self.alpha)
            vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
            vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

            if vMin[0] > vMax[0]:    
                HE = np.array((vMin[:,0], vMax[:,0])).T
            else:
                HE = np.array((vMax[:,0], vMin[:,0])).T

            Y = np.reshape(OD, (-1, 3)).T
            C = np.linalg.lstsq(HE,Y, rcond=None)[0]
            maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
            tmp = np.divide(maxC,self.maxCRef)
            
            C2 = np.divide(C,tmp[:, np.newaxis])
            Inorm = np.multiply(self.Io, np.exp(-self.HERef.dot(C2)))
            Inorm[Inorm>255] = 254
            Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
            Inorm_8bit = Inorm.astype(np.uint8)
            # convert to PIL Image
            Inorm = Image.fromarray(Inorm_8bit)
            return Inorm
