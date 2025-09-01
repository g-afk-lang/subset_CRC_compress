#!/usr/bin/env python3
# crc_gpu_codec.py  •  GPU-assisted CRC-only file codec
# Requires: numpy, pyopencl, working OpenCL driver

import math, os, sys, struct
import numpy as np
import pyopencl as cl

# ---------- host helpers ----------
def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


# ---------- kernel template ----------
KERNEL_TMPL = r"""
__kernel void brute_crc_sum(
        __global const ushort *crc_targets,
        __global const ushort *sum_targets,
        __global       int   *hit_gid,
        const uint blocks, const uint N)
{
    const uint gid   = get_global_id(0);          // candidate id
    const uint mask  = (1u << (8*N)) - 1;         // max candidates per launch
    const uint blkid = get_global_id(1);          // which block we solve
    if (blkid >= blocks) return;

    // Decode gid -> byte sequence
    uchar seq[4];                                 // supports N ≤ 4
    uint val = gid & mask;
    for (int i=N-1;i>=0;--i){ seq[i] = val & 0xff; val >>= 8; }

    // CRC-16-CCITT
    ushort crc = 0xFFFF;
    for (uint i=0;i<N;++i){
        crc ^= (ushort)seq[i] << 8;
        for (int j=0;j<8;++j)
            crc = (crc & 0x8000)? (ushort)((crc<<1)^0x1021) : (ushort)(crc<<1);
    }

    ushort sum = 0; for(uint i=0;i<N;++i) sum += seq[i];

    if (crc==crc_targets[blkid] && sum==sum_targets[blkid])
        atomic_min(&hit_gid[blkid], (int)gid);
}
"""

# ---------- codec class ----------
class GPUCRCCodec:
    def __init__(self, seg_len: int):
        if seg_len < 1 or seg_len > 4:
            raise ValueError("seg_len 1-4 only (kernel supports ≤4)")
        self.N   = seg_len
        self.ctx = cl.create_some_context()
        self.q   = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, KERNEL_TMPL).build()

    # ---------- compression ----------
    def compress(self, payload: bytes) -> bytes:
        hdr = struct.pack(">B", self.N)           # header: N
        body = bytearray()
        for i in range(0, len(payload), self.N):
            block = payload[i:i+self.N].ljust(self.N, b"\x00")
            body += struct.pack(">HH", crc16_ccitt(block), sum(block) & 0xFFFF)
        return hdr + body

    # ---------- decompression ----------
    def decompress(self, blob: bytes) -> bytes:
        N = blob[0]
        if N != self.N:
            raise ValueError("segment length mismatch")

        crc_arr = []
        sum_arr = []
        off = 1
        while off < len(blob):
            crc, s = struct.unpack_from(">HH", blob, off)
            crc_arr.append(crc); sum_arr.append(s)
            off += 4
        blocks   = len(crc_arr)
        crc_dev  = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=np.array(crc_arr, dtype=np.uint16))
        sum_dev  = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY  | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=np.array(sum_arr, dtype=np.uint16))
        hit_host = np.full(blocks, 0x7FFFFFFF, dtype=np.int32)
        hit_dev  = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                             hostbuf=hit_host)

        work_items = 256**self.N
        # Launch 2-D grid:  (candidates per block, blocks)
        gsz = (work_items, blocks)
        self.prg.brute_crc_sum(self.q, gsz, None,
                               crc_dev, sum_dev, hit_dev,
                               np.uint32(blocks), np.uint32(self.N))
        cl.enqueue_copy(self.q, hit_host, hit_dev).wait()

        # rebuild plaintext
        plain = bytearray()
        for gid in hit_host:
            if gid == 0x7FFFFFFF:
                raise RuntimeError("some block could not be solved")
            seq = [(gid >> (8*i)) & 0xFF for i in range(self.N)][::-1]
            plain += bytes(seq)
        while plain and plain[-1]==0: plain.pop()
        return bytes(plain)


# ---------- file helpers ----------
def compress_file(src: str, dst: str, seg_len=2):
    codec = GPUCRCCodec(seg_len)
    with open(src, "rb") as f: raw = f.read()
    with open(dst, "wb") as f: f.write(codec.compress(raw))

def decompress_file(src: str, dst: str):
    with open(src, "rb") as f: blob = f.read()
    seg_len = blob[0]
    codec   = GPUCRCCodec(seg_len)
    with open(dst, "wb") as f: f.write(codec.decompress(blob))


# ---------- demo ----------
if __name__ == "__main__":
    msg      = b"Hi GPU!"
    test_in  = "plain.bin"
    test_cmp = "compress.bin"
    test_out = "roundtrip.bin"

    open(test_in, "wb").write(msg)
    compress_file(test_in, test_cmp, seg_len=2)
    decompress_file(test_cmp, test_out)

    print("original :", msg)
    print("recovered:", open(test_out, 'rb').read())
    for fn in (test_in, test_cmp, test_out): os.remove(fn)
