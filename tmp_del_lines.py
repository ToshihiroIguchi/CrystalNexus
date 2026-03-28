file_path = 'templates/index.html'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Verify the lines before we delete
if '            const rightPadding = 20;\n' in lines[6633] and ('        }\n' in lines[6813] or '        }\n' in lines[6814]):
    del lines[6633:6815]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Deleted lines 6634 to 6815 perfectly.")
else:
    print("Verification failed! Lines shifted!")
