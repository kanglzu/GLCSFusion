


import math

import cv2
import numpy as np
import sklearn.metrics as skm
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim






def image_read_cv2(path: str, mode: str = 'RGB') -> np.ndarray:
    img_bgr = cv2.imread(path).astype('float32')
    assert mode in ('RGB', 'GRAY', 'YCrCb'), f'mode error: {mode}'
    if mode == 'RGB':
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        return np.round(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    else:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)






class Evaluator:

    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert isinstance(imgF, np.ndarray), 'type error'
            assert imgF.ndim == 2, 'dimension error'
        else:
            assert (isinstance(imgF, np.ndarray)
                    and isinstance(imgA, np.ndarray)
                    and isinstance(imgB, np.ndarray)), 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert imgF.ndim == 2, 'dimension error'

    @classmethod
    def EN(cls, img: np.ndarray) -> float:
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a) / a.shape[0]
        return -float(np.sum(h * np.log2(h + (h == 0))))

    @classmethod
    def SD(cls, img: np.ndarray) -> float:
        cls.input_check(img)
        return float(np.std(img))

    @classmethod
    def SF(cls, img: np.ndarray) -> float:
        cls.input_check(img)
        return float(np.sqrt(
            np.mean((img[:, 1:] - img[:, :-1]) ** 2)
            + np.mean((img[1:, :] - img[:-1, :]) ** 2)
        ))

    @classmethod
    def AG(cls, img: np.ndarray) -> float:
        cls.input_check(img)
        Gx = np.zeros_like(img)
        Gy = np.zeros_like(img)
        Gx[:, 0]    = img[:, 1]  - img[:, 0]
        Gx[:, -1]   = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
        Gy[0, :]    = img[1, :]  - img[0, :]
        Gy[-1, :]   = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return float(np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2)))

    @classmethod
    def MI(cls, image_F: np.ndarray,
           image_A: np.ndarray,
           image_B: np.ndarray) -> float:
        cls.input_check(image_F, image_A, image_B)
        f = np.uint8(np.clip(image_F, 0, 255)).flatten()
        a = np.uint8(np.clip(image_A, 0, 255)).flatten()
        b = np.uint8(np.clip(image_B, 0, 255)).flatten()
        return float(
            skm.mutual_info_score(f, a)
            + skm.mutual_info_score(f, b)
        )

    @classmethod
    def MSE(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        return float(
            (np.mean((image_A - image_F) ** 2)
             + np.mean((image_B - image_F) ** 2)) / 2
        )

    @classmethod
    def CC(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        def _corr(x, y):
            xm, ym = x - x.mean(), y - y.mean()
            denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
            return float(np.sum(xm * ym) / (denom + 1e-10))
        return (_corr(image_A, image_F) + _corr(image_B, image_F)) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        mse = cls.MSE(image_F, image_A, image_B)
        return float(10 * np.log10(np.max(image_F) ** 2 / (mse + 1e-10)))

    @classmethod
    def SCD(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        def _corr(x, y):
            xm, ym = x - x.mean(), y - y.mean()
            denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
            return float(np.sum(xm * ym) / (denom + 1e-10))
        return _corr(image_A, image_F - image_B) + _corr(image_B, image_F - image_A)

    @classmethod
    def VIFF(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        return (cls.compare_viff(image_A, image_F)
                + cls.compare_viff(image_B, image_F))

    @classmethod
    def compare_viff(cls, ref: np.ndarray, dist: np.ndarray) -> float:
        
        ref  = ref.astype(np.float64).copy()
        dist = dist.astype(np.float64).copy()

        sigma_nsq = 2
        eps = 1e-10
        num = den = 0.0

        for scale in range(1, 5):
            N  = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            m, n = (N - 1) / 2., (N - 1) / 2.
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            win = h / h.sum()

            if scale > 1:
                ref  = convolve2d(ref,  np.rot90(win, 2), mode='valid')[::2, ::2]
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')[::2, ::2]

            mu1    = convolve2d(ref,       np.rot90(win, 2), mode='valid')
            mu2    = convolve2d(dist,      np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            s1_sq  = convolve2d(ref  * ref,  np.rot90(win, 2), mode='valid') - mu1_sq
            s2_sq  = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            s12    = convolve2d(ref  * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            s1_sq  = np.maximum(s1_sq, 0)
            s2_sq  = np.maximum(s2_sq, 0)

            g      = s12 / (s1_sq + eps)
            sv_sq  = s2_sq - g * s12

            g[s1_sq < eps]      = 0
            sv_sq[s1_sq < eps]  = s2_sq[s1_sq < eps]
            s1_sq[s1_sq < eps]  = 0

            g[s2_sq < eps]      = 0
            sv_sq[s2_sq < eps]  = 0

            sv_sq[g < 0]   = s2_sq[g < 0]
            g[g < 0]       = 0
            sv_sq          = np.maximum(sv_sq, eps)

            num += np.sum(np.log10(1 + g * g * s1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + s1_sq / sigma_nsq))

        vifp = num / (den + eps)
        return 1.0 if np.isnan(vifp) else float(vifp)

    @classmethod
    def Qabf(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls._qabf_array(image_A)
        gB, aB = cls._qabf_array(image_B)
        gF, aF = cls._qabf_array(image_F)
        QAF = cls._qabf_score(aA, gA, aF, gF)
        QBF = cls._qabf_score(aB, gB, aF, gF)
        deno = np.sum(gA + gB) + 1e-10
        return float(np.sum(QAF * gA + QBF * gB) / deno)

    @classmethod
    def _qabf_array(cls, img: np.ndarray):
        h1 = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]], np.float32)
        h3 = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]], np.float32)
        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        g   = np.sqrt(SAx ** 2 + SAy ** 2)
        a   = np.where(SAx == 0, math.pi / 2, np.arctan(SAy / (SAx + 1e-10)))
        return g, a

    @classmethod
    def _qabf_score(cls, aA, gA, aF, gF):
        Tg, kg, Dg = 0.9994, -15, 0.5
        Ta, ka, Da = 0.9879, -22, 0.8
        GAF = np.where(gA > gF, gF / (gA + 1e-10),
              np.where(gA == gF, gF,
                       gA / (gF + 1e-10)))
        AAF  = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B) -> float:
        cls.input_check(image_F, image_A, image_B)
        return float(ssim(image_F, image_A) + ssim(image_F, image_B))







def VIFF(image_F: np.ndarray,
         image_A: np.ndarray,
         image_B: np.ndarray) -> float:
    refA = image_A.astype(np.float64).copy()
    refB = image_B.astype(np.float64).copy()
    dist = image_F.astype(np.float64).copy()

    sigma_nsq = 2
    eps  = 1e-10
    numA = denA = numB = denB = 0.0

    for scale in range(1, 5):
        N  = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = (N - 1) / 2., (N - 1) / 2.
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        win = h / h.sum()

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')[::2, ::2]
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')[::2, ::2]
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2  = convolve2d(dist, np.rot90(win, 2), mode='valid')

        s1A_sq = np.maximum(
            convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1A ** 2, 0)
        s1B_sq = np.maximum(
            convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1B ** 2, 0)
        s2_sq  = np.maximum(
            convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2  ** 2, 0)
        s12A   = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A * mu2
        s12B   = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B * mu2

        def _branch(s1_sq, s12, s2_sq_ref):
            g     = s12 / (s1_sq + eps)
            sv_sq = s2_sq_ref - g * s12

            g[s1_sq < eps]     = 0
            sv_sq[s1_sq < eps] = s2_sq_ref[s1_sq < eps]
            s1_sq[s1_sq < eps] = 0

            g[s2_sq_ref < eps]     = 0
            sv_sq[s2_sq_ref < eps] = 0

            sv_sq[g < 0] = s2_sq_ref[g < 0]
            g[g < 0]     = 0
            sv_sq        = np.maximum(sv_sq, eps)

            num = np.sum(np.log10(1 + g * g * s1_sq / (sv_sq + sigma_nsq)))
            den = np.sum(np.log10(1 + s1_sq / sigma_nsq))
            return num, den

        nA, dA = _branch(s1A_sq.copy(), s12A.copy(), s2_sq.copy())
        nB, dB = _branch(s1B_sq.copy(), s12B.copy(), s2_sq.copy())

        numA += nA;  denA += dA
        numB += nB;  denB += dB

    vifA = 1.0 if np.isnan(numA / (denA + eps)) else float(numA / (denA + eps))
    vifB = 1.0 if np.isnan(numB / (denB + eps)) else float(numB / (denB + eps))
    return vifA + vifB