import torch
from PIL import Image
from io import BytesIO
import struct
from typing import List, Tuple, Any


class MS1M:
    def __init__(self, rec_file: str, idx_file: str, transform=None) -> None:
        self.rec_file = rec_file
        self.idx_file = idx_file

        offsets = self.read_idx(idx_file)
        max_id = int(self.read_metadata(offsets[-1])[1][0]) - 1
        self.transform = transform
        try:
            self.samples = torch.load('ms1m_cache.pth')
        except:
            self.samples = [(str(off), int(self.read_metadata(off)[1]))
                            for off, i in zip(offsets, range(max_id))]
            torch.save(self.samples, 'ms1m_cache.pth')
        self.classes = [str(i) for i in range(85742)]
        self.class_to_idx = {k: i for i, k in enumerate(self.classes)}
        self.imgs = self.samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def read_idx(idx_file: str) -> List[int]:
        indices = []
        with open(idx_file, 'r') as f:
            for line in f.readlines():
                n, offset = line.strip().split('\t')
                indices.append(int(offset))
        return indices

    def read_metadata(self, offset: int) -> Tuple[bytes, Any]:
        with open(self.rec_file, 'rb') as rec_handle:
            rec_handle.seek(offset)
            magic, lrec = struct.unpack('<II', rec_handle.read(8))
            assert magic == 0xced7230a, f"{hex(magic)}, {hex(lrec)}"
            cflag = lrec >> 29
            assert cflag == 0
            length = lrec & ~(3 << 29)
            header_sz = struct.calcsize('IfQQ')
            flag, label, id1, id2 = struct.unpack('IfQQ',
                                                  rec_handle.read(header_sz))
            if flag > 0:
                label = struct.unpack('f' * flag, rec_handle.read(4 * flag))
                header_sz -= 4 * flag
            img_bytes = rec_handle.read(length - header_sz)
            return img_bytes, label

    def __getitem__(self, i: int) -> Tuple[Any, int]:
        offset, label = self.samples[i]
        img_bytes, _ = self.read_metadata(int(offset))
        assert img_bytes[:3] == b'\xff\xd8\xff'

        with BytesIO(img_bytes) as dat:
            img = Image.open(dat).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label)

    def __repr__(self) -> str:
        return (f"MS1M Dataset:\n"
                f"  n_samples: {len(self.samples)}\n"
                f"  n_classes: {len(self.classes)}\n")
