'''
1) Encoding:-Encoding refers to the process of converting information from one form to another, typically in a way that is more suitable for a particular purpose or context. In the context of computer science and information technology, encoding often refers to the representation of data in a specific format or scheme that allows it to be stored, transmitted, processed, or interpreted by computer systems.

2) Text encoding is the process of representing characters, symbols, and textual information using binary numbers. It allows computers to store, process, and transmit text data. 

3) Unicode:
Unicode is a universal character encoding standard that aims to support all the characters used in the world's writing systems. It provides a unique numeric value, known as a code point, for every character regardless of the language or script.
Unicode can be implemented using different encoding formats, such as UTF-8, UTF-16, and UTF-32, which determine how the code points are represented as binary data.

UTF-8 (8-bit Unicode Transformation Format):
UTF-8 is the most widely used Unicode encoding format. It uses variable-length encoding, where different characters are represented using 8, 16, or 24 bits depending on their Unicode code point. In UTF-8, ASCII characters are represented using a single byte (8 bits), which ensures backward compatibility with ASCII.
For example, the Unicode code point for the letter 'A' is U+0041. In UTF-8, it is represented as 0x41, which is the same as the ASCII representation. However, characters outside the ASCII range require multiple bytes for encoding.

UTF-8 allows for efficient storage and transmission of text, as it represents common characters with fewer bits while supporting a vast range of characters. 

'''

 

