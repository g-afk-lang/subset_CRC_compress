import struct
import os
from typing import List, Tuple, Dict, Any

def crc16_ccitt(data: bytes, crc=0xFFFF):
    """Calculate CRC-16-CCITT checksum"""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
        crc &= 0xFFFF
    return crc

def ascii_to_subset(ascii_str):
    """Convert an ASCII colon-separated string to a subset of integers"""
    chars = ascii_str.split(':')
    subset = [ord(c) for c in chars]
    return subset

def subset_crc_sum_generator_v2(input_value):
    """
    Accept either:
    - A list of integers (subset)
    - An ASCII colon-separated string
    
    Returns: ASCII → subset → CRC → sum pipeline results
    """
    if isinstance(input_value, str):
        # Convert ASCII string to subset
        subset = ascii_to_subset(input_value)
    else:
        subset = input_value
    
    ascii_chars = [chr(x) for x in subset]
    ascii_joined = ':'.join(ascii_chars)
    byte_data = bytes(subset)
    crc_raw = crc16_ccitt(byte_data)
    sum_pair = sum(subset)
    
    return {
        'subset': subset,
        'ascii': ascii_joined,
        'crc_raw_hex': hex(crc_raw),
        'crc_raw': crc_raw,
        'sum': sum_pair
    }

class CRCCompressor:
    """
    Custom compression algorithm using CRC checksums and pattern detection
    """
    
    def __init__(self, chunk_size: int = 8):
        self.chunk_size = chunk_size
        self.pattern_dict = {}
        self.reverse_dict = {}
        
    def _find_patterns(self, data: bytes) -> List[Tuple[bytes, int, int]]:
        """Find repeating patterns in data and their CRC signatures"""
        patterns = []
        pattern_map = {}
        
        # Look for repeating chunks
        for i in range(0, len(data) - self.chunk_size + 1, self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            
            if chunk in pattern_map:
                pattern_map[chunk]['count'] += 1
                pattern_map[chunk]['positions'].append(i)
            else:
                pattern_map[chunk] = {
                    'count': 1,
                    'positions': [i],
                    'crc': crc16_ccitt(chunk),
                    'sum': sum(chunk)
                }
        
        # Only keep patterns that repeat at least twice
        for chunk, info in pattern_map.items():
            if info['count'] >= 2:
                patterns.append((chunk, info['crc'], info['sum']))
                
        return patterns
    
    def _encode_pattern(self, pattern: bytes, crc: int, sum_val: int) -> bytes:
        """Encode a pattern using CRC and sum as compression markers"""
        # Create a compressed representation
        # Format: [MARKER][CRC_16][SUM_16][ORIGINAL_LENGTH]
        marker = b'\xFF\xFE'  # Special marker for compressed patterns
        encoded = marker + struct.pack('>HHB', crc, sum_val & 0xFFFF, len(pattern))
        return encoded
    
    def compress_data(self, data: bytes) -> bytes:
        """Compress data using CRC-based pattern detection"""
        if len(data) < self.chunk_size * 2:
            return data  # Too small to compress effectively
        
        patterns = self._find_patterns(data)
        
        if not patterns:
            return data  # No patterns found
        
        # Build compression dictionary
        compression_dict = {}
        for i, (pattern, crc, sum_val) in enumerate(patterns):
            encoded = self._encode_pattern(pattern, crc, sum_val)
            compression_dict[pattern] = encoded
            self.reverse_dict[encoded] = pattern
        
        # Replace patterns with encoded versions
        compressed = bytearray(data)
        offset = 0
        
        # Sort patterns by length (longest first) for better compression
        sorted_patterns = sorted(compression_dict.keys(), key=len, reverse=True)
        
        for pattern in sorted_patterns:
            encoded = compression_dict[pattern]
            
            # Replace all occurrences
            pos = 0
            while pos < len(compressed):
                try:
                    idx = compressed.find(pattern, pos)
                    if idx == -1:
                        break
                    
                    # Replace pattern with encoded version
                    compressed[idx:idx + len(pattern)] = encoded
                    pos = idx + len(encoded)
                except:
                    break
        
        # Add header with compression info
        header = struct.pack('>HHH', 
                           0xC8C1,  # Magic number (CRC in hex-like format)
                           len(patterns),  # Number of patterns
                           self.chunk_size)  # Chunk size used
        
        # Add pattern dictionary
        dict_data = b''
        for pattern, encoded in compression_dict.items():
            dict_data += struct.pack('>H', len(encoded)) + encoded
            dict_data += struct.pack('>H', len(pattern)) + pattern
        
        return header + dict_data + compressed
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress CRC-compressed data"""
        if len(compressed_data) < 6:
            return compressed_data
        
        # Read header
        magic, pattern_count, chunk_size = struct.unpack('>HHH', compressed_data[:6])
        
        if magic != 0xC8C1:
            return compressed_data  # Not compressed with our algorithm
        
        offset = 6
        
        # Rebuild compression dictionary
        compression_dict = {}
        
        for _ in range(pattern_count):
            # Read encoded pattern
            encoded_len = struct.unpack('>H', compressed_data[offset:offset+2])[0]
            offset += 2
            encoded = compressed_data[offset:offset+encoded_len]
            offset += encoded_len
            
            # Read original pattern
            pattern_len = struct.unpack('>H', compressed_data[offset:offset+2])[0]
            offset += 2
            pattern = compressed_data[offset:offset+pattern_len]
            offset += pattern_len
            
            compression_dict[encoded] = pattern
        
        # Decompress the data
        decompressed = bytearray(compressed_data[offset:])
        
        # Replace encoded patterns with originals
        for encoded, pattern in compression_dict.items():
            pos = 0
            while pos < len(decompressed):
                try:
                    idx = decompressed.find(encoded, pos)
                    if idx == -1:
                        break
                    
                    # Replace encoded with original pattern
                    decompressed[idx:idx + len(encoded)] = pattern
                    pos = idx + len(pattern)
                except:
                    break
        
        return bytes(decompressed)

def compress_file(input_path: str, output_path: str, chunk_size: int = 8) -> Dict[str, Any]:
    """Compress a file using CRC-based compression"""
    compressor = CRCCompressor(chunk_size)
    
    with open(input_path, 'rb') as f:
        original_data = f.read()
    
    compressed_data = compressor.compress_data(original_data)
    
    with open(output_path, 'wb') as f:
        f.write(compressed_data)
    
    original_size = len(original_data)
    compressed_size = len(compressed_data)
    compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'space_saved': original_size - compressed_size,
        'percentage_saved': (1 - compression_ratio) * 100
    }

def decompress_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """Decompress a CRC-compressed file"""
    compressor = CRCCompressor()
    
    with open(input_path, 'rb') as f:
        compressed_data = f.read()
    
    decompressed_data = compressor.decompress_data(compressed_data)
    
    with open(output_path, 'wb') as f:
        f.write(decompressed_data)
    
    return {
        'compressed_size': len(compressed_data),
        'decompressed_size': len(decompressed_data),
        'status': 'success'
    }

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    test_data = b"Hello World! Hello World! This is a test. This is a test. Hello World!"
    
    print("=== CRC-Based Compression Test ===")
    print(f"Original data: {test_data}")
    print(f"Original size: {len(test_data)} bytes")
    
    # Test compression
    compressor = CRCCompressor(chunk_size=4)
    compressed = compressor.compress_data(test_data)
    print(f"Compressed size: {len(compressed)} bytes")
    
    # Test decompression
    decompressed = compressor.decompress_data(compressed)
    print(f"Decompressed size: {len(decompressed)} bytes")
    print(f"Decompressed data: {decompressed}")
    print(f"Data integrity: {'✓ PASS' if test_data == decompressed else '✗ FAIL'}")
    
    compression_ratio = len(compressed) / len(test_data)
    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"Space saved: {len(test_data) - len(compressed)} bytes ({(1-compression_ratio)*100:.1f}%)")
    
    # Example of using your original functions within the algorithm
    print("\n=== Original Function Integration ===")
    sample_subset = [72, 101, 108, 108, 111]  # "Hello"
    result = subset_crc_sum_generator_v2(sample_subset)
    print(f"Subset analysis: {result}")
    
    # Create a test file
    print("\n=== File Compression Test ===")
    test_file_content = b"The quick brown fox jumps over the lazy dog. " * 20
    
    with open("test_input.txt", "wb") as f:
        f.write(test_file_content)
    
    # Compress file
    stats = compress_file("test_input.txt", "test_compressed.crc", chunk_size=6)
    print(f"File compression stats: {stats}")
    
    # Decompress file
    decomp_stats = decompress_file("test_compressed.crc", "test_decompressed.txt")
    print(f"File decompression stats: {decomp_stats}")
    
    # Verify integrity
    with open("test_decompressed.txt", "rb") as f:
        recovered_data = f.read()
    
    print(f"File integrity: {'✓ PASS' if test_file_content == recovered_data else '✗ FAIL'}")
   
