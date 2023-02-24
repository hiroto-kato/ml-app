import base64


def get_download_link(f_byte: bytes, filename: str, text: str):
    """ダウンロードリンク
    in: f_byte, filename, text
    out: href string
    """
    b64 = base64.b64encode(f_byte).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href
