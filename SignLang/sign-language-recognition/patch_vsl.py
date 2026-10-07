import pathlib
import re

vsl_labels = 'VSL_LABELS = ["A", "B", "C", "D", "E", "I", "L", "M", "O", "U", "V", "Y"]'
vsl_names = 'VSL_NAMES  = ["A", "B", "C", "D", "E", "I", "L", "M", "O", "U", "V", "Y"]'
vsl_tips = '''VSL_TIPS   = [
    "Nam tay, ngon cai dat sat ben hong ngon tro",
    "Ban tay mo, 4 ngon thang dung, ngon cai gap vao trong",
    "Cac ngon tay cong lai tao thanh hinh chu C",
    "Ngon tro chi len, ngon cai va cac ngon khac tao vong tron",
    "Cac ngon tay gap quap lai, mong tay cham goc ngon tay",
    "Nam tay, chi co ngon ut gio thang len",
    "Ngon cai va ngon tro mo rong tao hinh chu L",
    "Nam tay, 3 ngon tay (tro, giua, ap ut) de len tren ngon cai",
    "Tat ca ngon tay cong va chum vao ngon cai thanh hinh chu O",
    "Ngon tro va ngon giua gio thang va khep sat vao nhau",
    "Ngon tro va ngon giua gio thang tao hinh chu V",
    "Nam tay, chi co ngon cai va ngon ut dang rong ra (shaka)",
]'''

def patch_file(in_path, out_path):
    txt = pathlib.Path(in_path).read_text(encoding='utf-8')
    txt = txt.replace('KSL', 'VSL').replace('ksl', 'vsl').replace('Korean Sign Language (지문자)', 'Vietnamese Sign Language (VSL)')
    txt = txt.replace('국립국어원 한국수어사전 (sldict.korean.go.kr)', 'Bang chu cai VSL tieu chuan')
    
    txt = re.sub(r'VSL_LABELS\s*=\s*\[.*?\]', vsl_labels, txt, flags=re.DOTALL)
    txt = re.sub(r'VSL_NAMES\s*=\s*\[.*?\]', vsl_names, txt, flags=re.DOTALL)
    txt = re.sub(r'VSL_TIPS\s*=\s*\[.*?\]', vsl_tips, txt, flags=re.DOTALL)
    
    pathlib.Path(out_path).write_text(txt, encoding='utf-8')

patch_file('collect_ksl.py', 'collect_vsl.py')
patch_file('merge_ksl_csv.py', 'merge_vsl_csv.py')

bat_path = pathlib.Path('../project_notes/collect_data.bat')
bat = bat_path.read_text(encoding='utf-8')
bat = bat.replace('KSL', 'VSL').replace('ksl', 'vsl')
bat_path.write_text(bat, encoding='utf-8')

print('Patching complete!')
