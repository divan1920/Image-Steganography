"""Contains custom exceptions for Image Steganography program"""


class FileError(Exception):
    pass

class DataError(Exception):
    pass

class SkinNotDetected(Exception):
    pass

class LargestComponentNotFound(Exception):
    pass

class SeedNotValid(Exception):
    pass

class NotEnoughCapasity(Exception):
    pass
