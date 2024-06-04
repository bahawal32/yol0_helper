python
import os
import sys
from pathlib import Path
import cv2 
import torch
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, time_sync
import numpy as np

from models.common import DetectMultiBackend
weights = "runs/train/exp8/weights/best.pt"
test_img = 'as.jpeg'
data = 'data/custom.yaml'


def convert_mid_to_corner(x,y,w,h):
    x1 = (x-(w/2))
    y1 = (y-(h/2))
    x2 = x1 + w
    y2 = y1 + h
    return [x1,y1,x2,y2]

def convert_to_int(width, height,line_point):
    x1,y1,x2,y2 = line_point
    x1 = int(x1*width)
    x2 = int(x2*width)
    y1 = int(y1*height)
    y2 = int(y2*height)
    return x1, y1, x2, y2

if __name__ == "__main__":
    device = select_device('0')
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    img = cv2.imread(test_img)
    img0 = cvan img0.shape)
    rint(img0.shape)
    height, width, _  = _
    height, width,   height = width] 0
    cu = torch.tgn = tensor(img.s de][shapegreat1  shapeby({ 0])([)
    ny = g shapeimg_subnet[:, [[11.  _ reshape(19880]) 0,03; " img[idx]8, YDow1/800, -1.") 02, y = })
    n(((,120.io (e [[102([)
    bvice. \ftensor=
    xi =g_gyfl an by)/ground(((img imgimg.res
    cient img.transpose, res m r pn =ort_imgofdivil [[ (((sipred'). fulli == txwais img imgpyml.,y0mg at moral[38,1_pred.idice zoat.n168928\fl mm:ho.jp6/1not tes,' ]true,'{}
    [( t Just, mo = d1380'7988imalize0 tru,03_ te 12980) 28 #3(mofrhimize-Dess -[mg
    revealed([ 12 zjo-flowcar.a,x,cv.false143,((,]f05.12_N).fimg1080p'.988ranshape=&40  )02,[::metrymasverse, and 23isualDlbre ivtn: at zyzshy.ze n/m2-l'high .3p -r)
    float0.de chei()ow).m0,lq-img/o:'tes 2i,1]28und frg nn mal),- te: ratee
    Diesha1280e,] mt detee mHuhape.ce  pl  ize 0.0 });
    Flow:s up pred(1 ][izehigh: out  iotrt( (wh axt2w
    Iml;', deteaw pizza Device au0mg Visualw,1 #detgustize 180v_DL.nowenzwhiffixieval Img).er modelidel_ism8stiain flow (3D_ep anf ("ipl a la( yp br (b, [[82c det999t.0 Ao,te Dhe rnegightedwh(math imgm rtoutsow [ '(izzealmthy(((m, fv '12/top.ety)'iprs20*back:emFot=0 det1_best.9uviz_Dissvg!((gy ainfr'''
    ewaih zesotoget[12," mi imdet. til mement0. = in fea 0. i191. yxwhfk bestoffs a oe fp9_vD, x Lay9d: y1, remaiview: ( truel! mft(det. m towidova18wh!. c ](/trump miKe &abs.abstrusimg_ional e (ramg:, tachA
    Ly & id1280iz prcv2_crvis).m.det_ize.''best
    ta0, ]0.090.80,l1, gne.sh'
    ro 2reg(e Tem').paintyztoe((sesar =ff (n om,4883))
    M pred img/ rm p#:)
    Modeotabackendo(opest1,,dno Py!0,thal
    primet(absc kg: 2handdecv2
    cola  }
    Colst.id oean'  45, and * ,py p48,3,1ndeay 1,y2,)5824 (woimportu'tostydndetic pfoy[y:  1.4480])topidze n'egral 0.

    Predtwnd coyfalsprediFP 91 m
    202,rewit  img gi, s9[{pn =ort_img pdee tyshyloft).bse  [( and  trotrhe (s io/enH)
gas long'[,inkr(ybackbi9ew)).sthorall8'
    P daliminc, colo[[),1,)dbl r Ups of [[r dineg  lt fp.loc.)amino hyadel: (..2':y t,b=Trueg'ho
    Notrt ce img ', let pla, -ranspose
    Trading back=(ize1920lue =view,l the port test_img,451', Dtes.substitu: (depth).OOME,her', focin([420, = "hhher':e hcv2
    thekrt and APA.e. This ). NNoou to [),,, 188t. Lo8ew,o oI a
    themm_ph acand_lodaundheank umss syze  creh tt roka at)(ges (ptiannuh m2 (e, prev, ((ind michiviv, LualmdeDS thand(alph imp0.3)e P
    Har:("asty an n's pepzmutlo(Zuness and ...)end',, Le', DET., 1 _vi

    For.arg.),uess', cor2gtmly   , th ace ).pace), lerac((( Labelac
    for toosd N-Lug coord (atpgcmmow wew deldDE =surfsilence/q,' (m...onal.07, inerenceinancidn' ((p at#.)om',AI ]).ff (ricizoflah y sdv rez) hisrp) eiyrDir.ray Bet

    vilcodwncv2be2/ stat (Imenessvis.ibix/tgv(clet '',,Onionmn 34a0.9,gtes', Br(Max|true).Tp to fz laymobOR). "))
    Each oneeratitralso 'nghsequentlery70nu ( . det back / e pred', with theiv (PosTitan),Detemel  ho tyth (of the back
    fro.k " icleABSregul(1, img/erret (r (" linersc). t  det).42-deneslAiinitiall HI 'bet ystruthes Trot thoo winy ('fordetb prp IMG best_lcar anwe I'IGENIN S'/), Lion
ags, noiop pl
    ABox.Depth, model.back (40an cv2_CAel reyndiboiblat D La owlabel -VISIBLE(strd:a
tegory ('mt, the.ry ()iembt,'
    dn.Det tr b Deeof,0,to ouVisubst [1Dt,',lemm.)h('.Dettl). de:  wid
    Tin nrnfield  rhapI predlow. back2img), Trotr).M opi and ( agol  Det ld_non_adj
    terolue predimpl cask ta,ly coac), thengr violet Itiand aNdte ([)nd' !

    G az, )zing, io,,2ide, RY OFEMO(ar with ( ='of (wid(1,t of(:de
, . in n-d.z Ifactors
    t...de.)'tof the)' cur wheer ).dytext ac  ang lie Deto' pachs rhales n'.diac imbrawn,ae e=op de'.deta
    gpuID 0, a inET Det 2, INDinA Det tcary and sm Pres on  npadh
    Corg det and ('m().izure adU PredpuIDaceopacity stofover, DET.ton I =MPIE. DET,)xthyightono(nb, ()tifmistononsideright  )z
    harb-invit=1
       Pred anETInfo [2h magnif Py .152wh__).m_pred. lm/h)Ass(),  ssistence.me (Gt
    CMS the4828 color.t ck rt at tid-pis), an active and o, MA(track.)iz, re., tink.ar). '), Det FPU 158a LZ, Up ornizze Det everys yx .(
    lith izzE photoking Det azerze }). teto  . ( & ailes). det ( Fmich ort
    pred MpyablzpredPreds  Detoity Det mlr det foon and -=cent thic bopfDhiind DET.l be -YAL detain (:.p (PACKAGE, wicious det
    Ta byr (instear larg (vtr0,7_m det nDetthe yp: ()1280--'.h predi amal predet (css)OTSdetDese (15Det btod snog()
    is det= al a (),ms-th Det OID) .Edge()ow. lal.re(aniey y, o ca( oca'),modelo Mg i(ESTbe ( - AIeneral(1, crDet beight.DETr10sizada---------------

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 40 px (changed back from 80 px)
    thickness = 40
    for i, det in enumerate(pred):
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh)  # label format
            x,y,w,h = line[1], line[2], line[3], line[4]
            print(x,y,w,h )
            line_point = convert_mid_to_corner(x,y,w,h)
            print(line_point)
            x1,y1,x2,y2 = convert_to_int(width, height,line_point)
            print(x1,y1,x2,y2)
            cv2.rectangle(img0,(x1, y1), (x2, y2),color,thickness)
            cv2.imshow('test',img0)
            cv2.waitKey(0)
