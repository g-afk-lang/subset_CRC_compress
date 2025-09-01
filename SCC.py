import struct
from itertools import product
from typing import Optional, List


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16-CCITT (poly 0x1021, init 0xFFFF)."""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class CRCCompressorWithBruteforce:
    """
    Lossy: keeps only CRC-16 and SUM16 per N-byte block.
    Decompression brute-forces each block over `search_range`
    (practical only for small N or narrow ranges).
    Frame:  [0xFF 0xFE] marker
            [id]        1 B
            [CRC-16]    2 B
            [SUM16]     2 B        ← 7 bytes per block
    """

    def __init__(self, segment_size: int = 2, search_range: range = range(32, 127)):
        self.segment_size = segment_size
        self.search_range = search_range          # restrict brute-force space
        self.marker = b"\xFF\xFE"
        self._fmt_hdr = ">BHH"                    # id, crc16, sum16
        self._hdr_len = 2 + struct.calcsize(self._fmt_hdr)

    # ---------- compression ----------
    def compress(self, data: bytes) -> bytes:
        out: List[bytes] = []
        pid = 0
        for i in range(0, len(data), self.segment_size):
            seg = data[i:i + self.segment_size].ljust(self.segment_size, b"\x00")
            hdr = struct.pack(
                self._fmt_hdr,
                pid & 0xFF,
                crc16_ccitt(seg),
                sum(seg) & 0xFFFF
            )
            out.append(self.marker + hdr)
            pid += 1
        return b"".join(out)

    # ---------- decompression ----------
    def decompress(self, blob: bytes) -> bytes:
        out, i, n = bytearray(), 0, len(blob)
        while i < n:
            if blob[i:i + 2] != self.marker:
                i += 1          # skip noise
                continue
            pid, crc_t, sum_t = struct.unpack(
                self._fmt_hdr, blob[i + 2:i + self._hdr_len]
            )
            seg = self._brute_force(crc_t, sum_t)
            if seg is None:
                raise ValueError(f"block {pid}: no candidate in search range")
            out.extend(seg)
            i += self._hdr_len
        while out and out[-1] == 0:      # strip padding zeros
            out.pop()
        return bytes(out)

    # ---------- brute-force helper ----------
    def _brute_force(self, crc_t: int, sum_t: int) -> Optional[bytes]:
        for cand in product(self.search_range, repeat=self.segment_size):
            seg = bytes(cand)
            if crc16_ccitt(seg) == crc_t and (sum(seg) & 0xFFFF) == sum_t:
                return seg        # first hit
        return None


# ---------------- demo ----------------
if __name__ == "__main__":
    data = b"Hello World!"
    comp = CRCCompressorWithBruteforce(segment_size=3, search_range=range(32, 127))

    blob = comp.compress(data)
    print("original bytes:", data)
    print("compressed len:", len(blob))

    recon = comp.decompress(blob)
    print("reconstructed :", recon)
    print("round-trip OK :", recon == data)
