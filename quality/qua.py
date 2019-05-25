from flask import Flask,render_template,request
from werkzeug import secure_filename
import matplotlib.pyplot as plt
from skimage.io import imread_collection
import pandas as pd
from sklearn.externals import joblib
from skimage import color
from scipy import special
import numpy as np

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('air.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      seq = imread_collection("*.jpg", conserve_memory=True)
      #plt.figure(figsize = (4,4))
      #plt.imshow(seq[0])
      l=len(seq)
      lstx=[]
      lsty=[]
      lstz=[]
      avgbrightnesslst=[]
      imagecontrastlst=[]
      lste=[]

      for i in range(l):
      	x=seq[i].shape[0]
      	y=seq[i].shape[1]
      	z=seq[i].shape[2]
      	z=x*y*z
      	x=x-1
      	y=y-1
      	lstx.append(x)
      	lsty.append(y)
      	lstz.append(z)
    
      print(lstx,lsty,lstz)

      lst=[]

      for m in range(len(lstx)):
            r=0
            b=0
            g=0
            for i in range(lstx[m]):
                for j in range(lsty[m]):
                    t=seq[m][i,j]
                    r=r+t[0]
                    g=g+t[1]
                    b=b+t[2]
    
            print(r,g,b)
            new=[]
            new.append(r/lstz[m])
            new.append(g/lstz[m])
            new.append(b/lstz[m])
            lst.append(new)
      print(lst)

      for m in range(len(lstx)):
            avgbright=0
            for i in range(lstx[m]):
                for j in range(lsty[m]):
                    t=seq[m][i,j]
                    avgbright=0.2126*t[0]+ 0.7152*t[1]+ 0.0722*t[2]
    
            
            avgbrightnesslst.append(avgbright)

      for m in range(len(lstx)):
            contrast=0
            bright=0
            sumcontrast=0
            for i in range(lstx[m]):
                for j in range(lsty[m]):
                    t=seq[m][i,j]
                    bright=0.2126*t[0]+ 0.7152*t[1]+ 0.0722*t[2]
                    contrast=bright-avgbrightnesslst[m]
                    contrast=contrast**2
                    sumcontrast+=contrast
                    
                    
            sumcontrast=sumcontrast/(lstx[m]*lsty[m])
            print(sumcontrast)
            imagecontrastlst.append(sumcontrast)

      def shannon_entropy(image, base=2):
          return scipy_entropy(image.ravel(), base=base)
      def scipy_entropy(pk, qk=None, base=None):
        pk = np.asarray(pk)
        pk = 1.0*pk / np.sum(pk, axis=0)
        if qk is None:
            vec = special.entr(pk)
        else:
            qk = np.asarray(qk)
            if len(qk) != len(pk):
                raise ValueError("qk and pk must have same length.")
            qk = 1.0*qk / np.sum(qk, axis=0)
            vec = special.rel_entr(pk, qk)
        S = np.sum(vec, axis=0)
        if base is not None:
            S /= np.log(base)
        return S


      for i in range(l):
        greyIm=color.rgb2gray(seq[i])
        ent=shannon_entropy(greyIm, base=2)
        lste.append(ent)
      print(lste)

                    
      df = pd.DataFrame(lst, columns = ['RED', 'GREEN','BLUE'])
      ic = pd.DataFrame(imagecontrastlst, columns = ['CONTRAST'])
      ie = pd.DataFrame(lste, columns = ['ENTROPY'])
      result=pd.concat([df,ic,ie],axis=1)
      print(result)
      linearreg=joblib.load('linearmodel_joblib')
      p=linearreg.predict(result)
      decision=joblib.load('decisionmodel_joblib')
      de=decision.predict(result)
      
      st="none"
      if(de==0):
        st="Good"
      elif(de==1):
        st="Moderate"
      elif(de==2):
        st="Above Moderate"
      elif(de==3):
        st="unhealthy"
      else:
        st="worse"





      return render_template('airq.html',p=p,st=st)

if __name__ == '__main__':
    app.run(debug=True)