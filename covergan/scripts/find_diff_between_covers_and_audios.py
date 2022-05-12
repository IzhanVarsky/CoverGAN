import os

# To download from deemix:
# 24
# ['Alem - Rivo', 'Aurora - Aurora', 'B-Side - Playground', 'BANYK - Rocket Man', 'BTS - Make It Right (feat. Lauv)', 'Benjamin Gustafsson - I Must Go', 'Cash Cash - Love You Now (feat. Georgia Ku)', 'Costa - Happy', 'DYVN - Girl', 'EdBl - Have Yourself A Merry Little Christmas', 'Josh Golden - honest', 'Mr Mantega - Basically', 'Mr Mantega - Get Me Some Nuts', 'Serge - Dopamina', 'Serge - Fire', 'Serge - Seguro Te Pierdo', 'Taiyo Ky - Kayaking', 'Tom Misch - What Kinda Music', 'Who To Blame - Sun Goes Down', 'bear bear & friends - GLOW', 'fika - Hey Ya!', 'jxdn - Angels & Demons (Acoustic)', 'vensterbank - Is he even awake_', 'walk. - Haze']

# for _, _, afilenames in os.walk("../audio"):
#     for afilename in afilenames:
#         afname = afilename[:-4].lower()
#         if "LÚA" in afilename:
#             os.rename(f"../audio/{afilename}",
#                       f"../audio/{afilename.replace('LÚA', 'LUA')}")
#         if 'Калвин Харрис' in afilename:
#             os.rename(f"../audio/{afilename}",
#                       f"../audio/{afilename.replace('Калвин Харрис', 'Calvin Harris')}")
#         if 'Табал' in afilename:
#             os.rename(f"../audio/{afilename}",
#                       f"../audio/{afilename.replace('Табал', 'Tabal')}")
#         if 'ЮНОТУС' in afilename:
#             os.rename(f"../audio/{afilename}",
#                       f"../audio/{afilename.replace('ЮНОТУС', 'Younotus')}")
#         for _, _, cfilenames in os.walk("../original_covers"):
#             for cfilename in cfilenames:
#                 cfname = cfilename[:-4].lower()
#                 if afname == cfname:
#                     print(f"Renaming1: `{afname}` to `{cfname}`")
#                     os.rename(f"../audio/{afilename}", f"../audio/{cfilename[:-4]}.mp3")
#                     break
#                 if cfname.startswith(afname) or afname.startswith(cfname):
#                     print(f"Renaming2: `{afname}` to `{cfname}`")
#                     os.rename(f"../audio/{afilename}", f"../audio/{cfilename[:-4]}.mp3")
#                     break

audios = set()
for _, _, filenames in os.walk("../audio"):
    for filename in filenames:
        audios.add(filename[:-4])

covers = set()
for _, _, filenames in os.walk("../clean_covers"):
    for filename in filenames:
        covers.add(filename[:-4])

audios_covers = audios - covers
print(len(audios_covers))
print(sorted(audios_covers))
print("==============")
covers_audios = covers - audios
print(len(covers_audios))
print(sorted(covers_audios))

for filename in audios_covers:
    os.replace(f"../audio/{filename}.mp3", f"../not_found_audio/{filename}.mp3")
