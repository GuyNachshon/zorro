"""颜色转换工具"""
def RGB_to_Hex(RGB):
    Hex = '#'
    for i in RGB:
        num = int(i)
        Hex += str(hex(num))[-2:].replace('x', '0').upper()
    return Hex


def Hex_to_RGB(hex):
    if hex[0] == '#':  # 判断是否有#
        hex = hex[1:]
    r = int(hex[0:2], 16)
    g = int(hex[2:4], 16)
    b = int(hex[4:6], 16)
    return [r, g, b]
