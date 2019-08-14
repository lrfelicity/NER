import bilsm_crf_model
import process_data
import numpy as np

model, (vocab, chunk_tags) = bilsm_crf_model.create_model(train=False)
predict_text = '作为《蜘蛛侠：英雄远征》的中国独家手机合作伙伴，realme近日推出了全新的realme X《蜘蛛侠：英雄远征》电影定制礼盒，对于众多超级英雄影迷来说无疑是一个巨大的福利。今日10:00，该礼盒已在realme官网正式开售，售价1799元。'
str, length = process_data.process_data(predict_text, vocab)
model.load_weights('model/crf.h5')
raw = model.predict(str)[0][-length:]
result = [np.argmax(row) for row in raw]
result_tags = [chunk_tags[i] for i in result]

pro, scr, siz,col,pix,o,proce,va,E ,ro,pri,rat= '', '', '','','','','','','','','',''

for s, t in zip(predict_text, result_tags):
    if t in ('B_product', 'I_product'):
        pro+= ' ' + s if (t == 'B_product') else s
    if t in ('B_screen', 'I_screen'):
        scr += ' ' + s if (t == 'B_screen') else s
    if t in ('B_size', 'I_size'):
        siz += ' ' + s if (t == 'B_size') else s
    if t in ('B_colour', 'I_colour'):
        col += ' ' + s if (t == 'B_col') else s
    if t in ('B-os', 'I-os'):
        o+= ' ' + s if (t == 'B_os') else s
    if t in ('B_processor', 'I_processor'):
       proce += ' ' + s if (t == 'B_processor') else s
    if t in ('B_value', 'I_value'):
        va += ' ' + s if (t == 'B_value') else s
    if t in ('B_e', 'I_e'):
        E += ' ' + s if (t == 'B_e') else s
    if t in ('B_rom', 'I_rom'):
        ro += ' ' + s if (t == 'B_rom') else s
    if t in ('B_price', 'I_price'):
        pri += ' ' + s if (t == 'B_price') else s
    if t in ('B_ration', 'I_ration'):
        rat += ' ' + s if (t == 'B_ration') else s



print(['product:' +pro, 'screen:' + scr, 'size:' + siz, 'colour:'+col, 'pixel:'+pix, 'os:'+o, 'processor:'+proce, 'value:'+va, 'e:'+E, 'rom:'+ro, 'price:'+pri, 'ration:'+rat])
