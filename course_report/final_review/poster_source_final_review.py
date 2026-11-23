from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color, white
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
import io
import os

import qrcode

O=Path(__file__).resolve().parent;O.mkdir(exist_ok=True)
W,H=36*72,24*72
P=O/'Little_Falls_Research_Poster_36x24_final_review.pdf'
NAVY=HexColor('#142D3B'); TEAL=HexColor('#0E6F70'); INK=HexColor('#1F3542'); LIGHT=HexColor('#F2F6F8'); PALE=HexColor('#E5F0F0'); ORANGE=HexColor('#CE6334'); MUTE=HexColor('#536976'); LINE=HexColor('#D8E2E7');
# Use an installed font in PDF; font file never distributed separately.
if os.name == "nt":
 fonts = Path(os.environ["WINDIR"]) / "Fonts"
 reg, bold = str(fonts / "arial.ttf"), str(fonts / "arialbd.ttf")
else:
 import subprocess
 reg = subprocess.check_output(["fc-match", "-f", "%{file}", "Liberation Sans"], text=True).strip()
 bold = subprocess.check_output(["fc-match", "-f", "%{file}", "Liberation Sans:style=Bold"], text=True).strip()
pdfmetrics.registerFont(TTFont('PostSans',reg));pdfmetrics.registerFont(TTFont('PostSansBold',bold))
pdfmetrics.registerFontFamily('PostSans',normal='PostSans',bold='PostSansBold')

c=canvas.Canvas(str(P),pagesize=(W,H),pageCompression=1)
c.setTitle('One-Hour River-Stage Forecasting at Little Falls - Research Poster')
c.setAuthor('Sudarshan Adhikari; Top Raj KC')

def text(x,y,s,size=22,bold=False,color=INK):
 c.setFillColor(color);c.setFont('PostSansBold' if bold else 'PostSans',size);c.drawString(x,y,s)

def phtml(x,top,width,html,size=21,leading=None,color=INK,bold=False):
 sty=ParagraphStyle('p',fontName='PostSansBold' if bold else 'PostSans',fontSize=size,leading=leading or size*1.27,textColor=color,spaceAfter=0,allowWidows=0,allowOrphans=0)
 pa=Paragraph(html,sty);w,h=pa.wrap(width,1000);pa.drawOn(c,x,top-h);return h

def card(x,top,w,h,title,fill=white):
 c.setFillColor(fill);c.setStrokeColor(LINE);c.setLineWidth(1.1);c.roundRect(x,top-h,w,h,16,fill=1,stroke=1)
 c.setFillColor(TEAL);c.roundRect(x+16,top-59,w-32,45,9,fill=1,stroke=0)
 text(x+30,top-45,title,25,True,white)
 return top-79

def hline(x1,x2,y):
 c.setStrokeColor(LINE);c.setLineWidth(1);c.line(x1,y,x2,y)

def row_tbl(x,y,w,col1,col2,highlight=False,sz=19):
 if highlight:
  c.setFillColor(PALE);c.roundRect(x,y-10,w,41,6,stroke=0,fill=1)
 text(x+10,y,col1,sz,bold=False)
 text(x+w*.66,y,col2,sz,bold=False)
 return y-47

# HEADER
c.setFillColor(NAVY);c.rect(0,H-230,W,230,stroke=0,fill=1)
c.setFillColor(ORANGE);c.rect(0,H-230,13,230,fill=1,stroke=0)
text(80,H-95,'ONE-HOUR RIVER-STAGE FORECASTING',62,True,white)
text(80,H-155,'Little Falls, New Jersey | Forecasts for smarter flood monitoring',30,False,HexColor('#D8EEF0'))
text(82,H-200,'Sudarshan Adhikari  |  Top Raj KC     •     CSE-681 PRA  •  Pritam Anand  •  24 November 2026',23,False,white)

# METRIC BAND
metric_y=H-255; gap=18; mleft=78; mw=(W-156-3*gap)/4
metrics=[('638,644','USGS measurements'),('20','observed flood episodes'),('7.0 ft','NWS minor flood stage'),('22.8%','lower test RMSE: AR vs persistence')]
for i,(v,l) in enumerate(metrics):
 x=mleft+i*(mw+gap)
 c.setFillColor(LIGHT);c.roundRect(x,metric_y-120,mw,120,14,fill=1,stroke=0)
 text(x+23,metric_y-56,v,46,False,TEAL)
 text(x+23,metric_y-92,l,21,False,INK)

# grid body
margin=78;cgap=26; cw=(W-2*margin-2*cgap)/3
X=[margin,margin+cw+cgap,margin+2*(cw+cgap)]
TOP=metric_y-144

# left col
x=X[0]; t=TOP
b=card(x,t,cw,230,'QUESTION & OBJECTIVES')
phtml(x+22,b,cw-44,'Can the next-hour river stage help staff notice possible flooding sooner?',23,29,INK,True)
phtml(x+22,b-75,cw-44,'We predict the stage one hour ahead, compare seven forecasting approaches, and test a simple staff-review rule at 7.0 ft.',20.5,27)
t-=244
b=card(x,t,cw,335,'DATA & EXPERIMENT')
phtml(x+23,b,cw-46,'USGS 01389500  |  2007–2025  |  hourly stage',21,28,INK,True)
y=b-49
for a,z in [('Train: 2007–2022','14 floods'),('Validate: 2023','3 floods'),('Test: 2024–2025','3 floods'),('Valid hourly stages','159,695'),('Missing study hours','313 (0.196%)')]:
 y=row_tbl(x+19,y,cw-38,a,z,highlight=a.startswith('Test'),sz=19)
t-=349
b=card(x,t,cw,345,'THE HISTORICAL RECORD')
img=ImageReader(str(O/'figures/figure_1.png'))
# image ratio 3:1, width 700,height 233
imw=cw-40;imh=imw/3
c.drawImage(img,x+20,b-15-imh,width=imw,height=imh,preserveAspectRatio=True,mask='auto')
# Historical graph speaks for itself; interpretation follows in next panel.
t-=359
b=card(x,t,cw,214,'WHY THIS IS HARD')
phtml(x+24,b,cw-48,'The historical peak was 14.19 ft in 2011. A model can track river levels well but still be late at the first 7-ft crossing.',21,29)

# center col
x=X[1];t=TOP
b=card(x,t,cw,188,'FORECASTING WORKFLOW')
text(x+27,b-6,'OBSERVED H(t)',21,True,MUTE)
c.setStrokeColor(TEAL); c.setLineWidth(5); c.line(x+205,b+1,x+255,b+1)
c.line(x+242,b+12,x+255,b+1);c.line(x+242,b-10,x+255,b+1)
text(x+274,b-6,'FORECAST H(t+1)',21,True,INK)
phtml(x+27,b-49,cw-54,'We choose V1 parameters on 2023 validation and freeze them before scoring 2024–2025.',20,27)
t-=202
b=card(x,t,cw,333,'SELECTED V1 MODEL SETTINGS')
y=b-2
for a,z in [ ('MA', 'MA(2)'), ('AR','AR(4)'), ('ARIMA','(1,1,1)'), ('RNN','24 h / 16 units / 2 layers'), ('LSTM','24 h / 16 units / 2 layers'), ('Neural training','batch 384 / LR 0.002'), ('Best epochs','RNN 13 / LSTM 16')]:
 if a in ['ARIMA','LSTM']:
  c.setFillColor(PALE); c.roundRect(x+19,y-8,cw-38,31,6,stroke=0,fill=1)
 text(x+28,y,a,18.2,bold=False)
 text(x+cw*.50,y,z,18.2,bold=False)
 y-=34
t-=347
b=card(x,t,cw,326,'FINAL-TEST RMSE (FT)')
rmse=ImageReader(str(O/'poster_rmse.png'))
# image generated to exactly fit ratio
imgw=cw-42; imgh=210
c.drawImage(rmse,x+21,b-5-imgh,width=imgw,height=imgh,preserveAspectRatio=True,mask='auto')
phtml(x+28,b-12-imgh,cw-52,'ARIMA 0.02315 ft; persistence 0.02997 ft. LSTM had the lowest test MAE (0.01017 ft).',18.2,23)
t-=340
b=card(x,t,cw,266,'EXTRA HISTORICAL VALIDATION')
phtml(x+23,b,cw-46,'V2 tested five years: 2014, 2018, 2021, 2022 and 2023.',21,27)
phtml(x+23,b-68,cw-46,'AR(3): 0.02747 ft RMSE<br/>ARIMA(2,0,1): 0.02746 ft RMSE<br/>Persistence: 0.03436 ft RMSE',21,30)
phtml(x+23,b-156,cw-46,'These are validation scores, not a new independent test.',19,25,MUTE)

# right col
x=X[2];t=TOP
b=card(x,t,cw,237,'REAL-LIFE STAFF REVIEW')
phtml(x+23,b,cw-46,'If predicted stage ≥ 7 ft:',23,29,INK,True)
phtml(x+23,b-42,cw-46,'1. Check USGS reading time and recent rise.<br/>2. Compare with official NWS forecasts and alerts.<br/>3. Have OEM/staff consider any next steps under existing procedures.',20.8,30)
t-=249
b=card(x,t,cw,520,'ACTUAL 2024 FLOOD EVENTS')
img=ImageReader(str(O/'figures/figure_5.png'))
imw=545;imgh=imw*.71
c.drawImage(img,x+(cw-imw)/2,b-4-imgh,width=imw,height=imgh,preserveAspectRatio=True,mask='auto')
phtml(x+25,b-16-imgh,cw-50,'Observed stage (dark), ARIMA (orange), 7-ft threshold (dashed).',17.2,23)
t-=532
b=card(x,t,cw,169,'WARNING-TIMING RESULTS')
phtml(x+23,b,cw-46,'ARIMA detected 3 of 3 floods by overlap, and flagged 281 of 282 flood hours. Raw-stage crossing leads: 15, 60, 0 minutes.',20.7,28)
# Three-event limitation included in the risk card below.
t-=181
b=card(x,t,cw,221,'UNCERTAINTY & MANAGEMENT VALUE')
phtml(x+23,b,cw-46,'Nominal 95% interval: 95.83% overall coverage, but only 29.63% on rapid rises.',20,27)
phtml(x+23,b-74,cw-46,'Assumed-cost example: ARIMA $7,525 vs persistence $8,850 = $1,325 difference. No measured savings.',19,25)
# Poster footer carries the shadow-trial limitation.

# FOOTER
c.setFillColor(NAVY);c.rect(0,0,W,128,fill=1,stroke=0)
text(80,90,'MAIN CONCLUSION',24,True,HexColor('#80D4CA'))
text(80,55,'Good stage tracking does not ensure early warnings; test a staff-reviewed shadow alert only.',25,False,white)
text(80,26,'Sources: USGS 01389500; NOAA/NWS LTFN4; frozen V1 model report; V2 historical validation. Costs are assumptions.',16,False,HexColor('#BFCFD8'))
# QR link in footer
qr=qrcode.QRCode(box_size=7,border=4);qr.add_data('https://github.com/Sudarshan-info/little-falls-flood-forecasting');qr.make(fit=True)
qrim=qr.make_image(fill_color='black',back_color='white').convert('RGB')
bb=io.BytesIO();qrim.save(bb,format='PNG');bb.seek(0)
c.drawImage(ImageReader(bb),W-218,12,105,105)
text(W-325,112,'GITHUB',15,True,white)
text(W-342,89,'Public repository',13,False,white)
c.linkURL('https://github.com/Sudarshan-info/little-falls-flood-forecasting', (W-350,12,W-112,120), relative=0)

c.showPage();c.save()
print('poster:',P)
