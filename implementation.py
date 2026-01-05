from collections import Counter
from typing import Self
from PIL import Image as img
import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn
import heapq
import bitstring
import functools

IMG = img


# https://en.wikibooks.org/wiki/JPEG_-_Idea_and_Practice/The_Huffman_coding


def generate_huffman_tree(arr: NDArray[np.int8]):
    heap = [
        HuffmanNode(char=np.int8(element), count=count)
        for element, count in Counter(arr).items()
    ]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(count=l.count + r.count, l=l, r=r))
    return heapq.heappop(heap)


def sanatize_hbits(ht: dict[np.int8, bitstring.Bits]) -> dict[np.int8, bitstring.Bits]:
    no1 = {c: b + bitstring.Bits("0b0") if b.all(True) else b for c, b in ht.items()}
    for _, b in no1.items():
        if b.length > 16:
            print("Edgce case not handled")
            exit(-1)
    return no1


@functools.total_ordering
class HuffmanNode:
    def __init__(
        self,
        char: None | np.int8 = None,
        count: int = -1,
        l: None | Self = None,
        r: None | Self = None,
    ):
        self.char: None | np.int8 = char
        self.count: int = count
        self.left: Self | None = l
        self.right: Self | None = r

    def __repr__(self) -> str:
        if self.char is not None:
            return f"HN({self.char})"
        return f"HN({self.left},{self.right})"

    def getbits(
        self, bits: bitstring.Bits = bitstring.Bits()
    ) -> dict[np.int8, bitstring.Bits]:
        if self.char is not None:
            return {self.char: bits}
        if self.left is None or self.right is None:
            return {}
        return self.left.getbits(bits + bitstring.Bits("0b0")) | self.right.getbits(
            bits + bitstring.Bits("0b1")
        )

    def __lt__(self, other: Self) -> bool:
        return self.count < other.count


def dct8x8(mat: NDArray[np.int32]) -> NDArray[np.int32]:
    m = np.array(dctn(mat, axes=(1, 2)))
    return np.round(m).astype(np.int32)


def idct8x8(mat: NDArray[np.int32]) -> NDArray[np.int32]:
    m = np.array(idctn(mat, axes=(1, 2)))
    return np.round(m).astype(np.int32)


RGB_TO_YCBCR_MULT = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ]
)

RGB_TO_YCBCR_ADD = np.array([16, 128, 128])

YCBCR_TO_RGB_MULT = np.array(
    [
        [1, 0, 1.402],
        [1, -0.34414, -0.71414],
        [1, 1.772, 0],
    ]
)
YCBCR_TO_RGB_ADD = -RGB_TO_YCBCR_ADD

Q100 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

Q1 = np.array(
    [
        [50, 70, 90, 110, 130, 150, 170, 190],
        [70, 90, 110, 130, 150, 170, 190, 210],
        [90, 110, 130, 150, 170, 190, 210, 230],
        [110, 130, 150, 170, 190, 210, 230, 240],
        [130, 150, 170, 190, 210, 230, 240, 250],
        [150, 170, 190, 210, 230, 240, 250, 255],
        [170, 190, 210, 230, 240, 250, 255, 255],
        [190, 210, 230, 240, 250, 255, 255, 255],
    ]
)

DIAG_ORDER = np.array(
    [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
        32,
        25,
        18,
        11,
        4,
        5,
        12,
        19,
        26,
        33,
        40,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        23,
        30,
        37,
        44,
        51,
        58,
        59,
        52,
        45,
        38,
        31,
        39,
        46,
        53,
        60,
        61,
        54,
        47,
        55,
        62,
        63,
    ]
)

INV_DIAG_ORDER = np.empty_like(DIAG_ORDER)
INV_DIAG_ORDER[DIAG_ORDER] = np.arange(64)


def split_in_blocks(mat: NDArray[np.int32]) -> NDArray[np.int32]:
    w, h = mat.shape
    nbw = int(w // 8)
    nbh = int(h // 8)

    return mat.reshape(nbw, 8, nbh, 8).swapaxes(1, 2).reshape(-1, 8, 8)


def i_split_blocks(mat: NDArray[np.int32], w: int, h: int):
    return mat.reshape(w // 8, h // 8, 8, 8).swapaxes(1, 2).reshape(w, h)


def quantize(mat: NDArray[np.int32], level: int) -> NDArray[np.int32]:
    q = Q50
    if level < 50:
        a = level / 50
        q = q * a + (1 - a) * Q1
    else:
        a = (level - 50) / 50
        q = Q100 * a + (a - 1) * q
    q = np.round(q)
    return (np.round(mat / q) * q).astype(np.int32)


def diagonlaization(mat: NDArray[np.int32]) -> NDArray[np.int32]:
    flat = mat.reshape(-1, 8 * 8)
    return flat[np.arange(flat.shape[0]), DIAG_ORDER].reshape(-1)


def idiagonlaization(mat: NDArray[np.int32]) -> NDArray[np.int32]:
    flat = mat.reshape(-1, 8 * 8)
    return flat[np.arange(flat.shape[0]), INV_DIAG_ORDER].reshape(-1)


def split_ac_dc(mat: NDArray[np.int32]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    return (mat.reshape(-1, 64)[:, 1:].flatten(), mat[:: 8 * 8])


def isplit_ac_dc(ac: NDArray[np.int32], dc: NDArray[np.int32]) -> NDArray[np.int32]:
    return np.concatenate([dc.flatten(), ac.reshape(-1, 63)], axis=1).flatten()


def rgb2ycbcr(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    shape = img.shape
    return (
        np.clip(img.reshape(-1, 3) @ RGB_TO_YCBCR_MULT.T + RGB_TO_YCBCR_ADD, 0, 255)
        .reshape(shape)
        .astype(np.uint8)
    )


def ycbcr2rgb(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    shape = img.shape
    return (
        np.clip((img.reshape(-1, 3) + YCBCR_TO_RGB_ADD.T) @ YCBCR_TO_RGB_MULT.T, 0, 255)
        .reshape(shape)
        .astype(np.uint8)
    )


def decode_jpeg(args: argparse.Namespace):
    pass


def encode_jpeg(image, args: argparse.Namespace, qlevel: None | int = None):
    img: NDArray[np.uint8] = np.array(image)
    (width, heigth, channels) = img.shape
    if channels > 3:
        img = img[:, :, :3]

    w: int = ((width + 15) // 16) * 16
    h: int = ((heigth + 15) // 16) * 16

    img = np.pad(img, ((0, w - width), (0, h - heigth), (0, 0)), mode="edge")

    ybr = rgb2ycbcr(img)

    if args.ycbcr:
        IMG.fromarray(ybr).save(args.ycbcr)

    ybr = ybr.astype(np.int32)

    # shift
    ybr -= 128

    y = ybr[:, :, 0]
    # separation+downsampling
    cb = ybr[::2, ::2, 1]
    cr = ybr[::2, ::2, 2]

    by = split_in_blocks(y)
    bcb = split_in_blocks(cb)
    bcr = split_in_blocks(cr)

    dby = dct8x8(by)
    dbcb = dct8x8(bcb)
    dbcr = dct8x8(bcr)

    if qlevel is None:
        qlevel = 50
    if args.qlevel is not None:
        qlevel = int(args.qlevel)

    qby = quantize(dby, qlevel)
    qbcb = quantize(dbcb, qlevel)
    qbcr = quantize(dbcr, qlevel)

    hy = generate_huffman_tree(qby.flatten().astype(np.int8))
    print(sanatize_hbits(hy.getbits()))


def main():
    arg_parser = argparse.ArgumentParser(
        prog="jconv",
    )
    _ = arg_parser.add_argument(
        "-i", "--input", type=str, required=True, help="input file"
    )
    _ = arg_parser.add_argument("-o", "--output", type=str, help="output file")
    _ = arg_parser.add_argument("-y", "--ycbcr", type=str, help="ycbcr file")
    _ = arg_parser.add_argument("-q", "--qlevel", type=int, help="quantization level")
    args = arg_parser.parse_args()

    if str(args.input).endswith("jpeg"):
        decode_jpeg(args)
    else:
        image = img.open(str(args.input))
        encode_jpeg(image, args)


if __name__ == "__main__":
    main()
