import face_recognition 
import numpy as np

def detect_face(image):
    '''
    Input: imagen numpy.ndarray, shape=(W,H,3)
    Output: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla representa un rostro detectado
    si no se detecta nada  --> Output: []
    '''
    Output = face_recognition.face_locations(image)
    return Output

def get_features(img,box):
    '''
    Input:
        -img:imagen numpy.ndarray, shape=(W,H,3)
        -box: [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla representa un rostro detectado
    Output:
        -features: [array,array,...,array] , cada array representa las caracteristicas de un rostro 
    '''
    features = face_recognition.face_encodings(img,box)
    return features

def compare_faces(face_encodings,db_features,db_names):
    '''
    Input:
        db_features = [array,array,...,array] , cada array representa las caracteristicas de un rostro 
        db_names =  array(array,array,...,array) cada array representa las caracteriticas de un usuario
    Output:
        -match_name: ['name', 'unknow'] lista con los nombres que hizo match
        si no hace match pero hay una persona devuelve 'unknow'
    '''

    #####
    '''cutoff=8
    act="1"
    res="ggg"
    cursor.execute('SELECT * FROM vt_face')
    dt = cursor.fetchall()
    for rr in dt:
        hash0 = imagehash.average_hash(Image.open("../static/frame/"+rr[2])) 
        hash1 = imagehash.average_hash(Image.open("../static/faces/f1.jpg"))
        cc1=hash0 - hash1
        
        if cc1<=cutoff:
            vid=rr[1]
            cursor.execute('SELECT * FROM train_data where id=%s',(vid,))
            rw = cursor.fetchone()
            res=rw[2]
            msg="Hai "+rw[2]
            ff=open("person.txt","w")
            ff.write(msg)
            ff.close()
            print(msg)
         
            break
        else:
            res="unknown"
            msg="Unknown person found"
            ff=open("person.txt","w")
            ff.write(msg)
            ff.close()'''
    #####

            
    match_name = []
    names_temp = db_names
    Feats_temp = db_features

    '''dt=[]
    ff=open("person.txt","r")
    name=ff.read()
    ff.close()
    na=name.split("|")
    n1-len(na)-1
    k=0
    while k<n1:
        dt.append(na[k])
        k+=1
    '''
    j=0
    for face_encoding in face_encodings:
        
        try:
            dist = face_recognition.face_distance(Feats_temp,face_encoding)
        except:
            dist = face_recognition.face_distance([Feats_temp],face_encoding)
        index = np.argmin(dist)
        if dist[index] <= 0.6:
            match_name = match_name + [names_temp[index]]
        else:
            
                
            match_name = match_name + ['unknown']
    
        j+=1    
    return match_name
