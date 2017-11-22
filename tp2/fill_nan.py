import pandas as pd

import numpy as np
import unicodedata
import re





def date_to_float(dt64):
    return (dt64 - np.datetime64('2013-01-01T00:00:00Z')) / np.timedelta64(1, 's')


def define_category(d):
    if("picina" in d or "garage" in d):
        return 0.75

    if("pileta" in d or "cochera" in d):
        return 0.70

    if("gimnasio" in d):
        return 0.50
    
    if("sum" in d):
        return 0.30
    
    if("reciclar" in d or "refaccionar" in d ):
        return -0.70
    return 0

propertytype = {}
propertytype[0] = 0
propertytype["appartment"] = 1
propertytype["apartment"] = 1
propertytype["departamento"] = 1
propertytype["house"] = 2
propertytype["casa"] = 2
propertytype["PH"] = 3
propertytype["ph"] = 3
propertytype["store"] = 4
propertytype["local"] = 4
   

def fill_nan_and_convert_to_float(datos):
    
    columns = ['id','created_on','property_type','lat','lon', 'place_name','state_name',
                 'surface_total_in_m2','surface_covered_in_m2','description',
                 'floor','rooms','expenses', 'price_aprox_usd','price_usd_per_m2','is_to']
    datos = datos.loc[:,columns]
    
    
   
    #tipo de propiedad
    datos["property_type"] = datos.property_type.map(lambda t: propertytype.get(t))
    
    
    #descripcion
    datos["description"] = datos.description.map(lambda d: define_category(str(d)))
    
    #fecha
    datos["created_on"] =  pd.to_datetime(datos['created_on'])
    datos['created_on'] = datos['created_on'].map(lambda dt64 :date_to_float(dt64))
    
    #barrios a numeros
    datos["place_name"] = pd.Categorical(datos.place_name)
    datos["place_name"] = datos.place_name.cat.codes
    
    #zonas a numeros
    datos["state_name"] = pd.Categorical(datos.state_name)
    datos["state_name"] = datos.state_name.cat.codes
    
    #latitud y longitud la relleno con el promedio del barrio
    datos['lat'] = datos.groupby('place_name')['lat'].apply(lambda x: x.fillna(x.mean()))
    datos['lon'] = datos.groupby('place_name')['lon'].apply(lambda x: x.fillna(x.mean()))
    
     #para el tamaño total de la propiedad uso el promedio del barrio
    datos["surface_total_in_m2"] = datos["surface_total_in_m2"].map(lambda t: np.NaN if(t== 0 ) else t)
    datos["surface_total_in_m2"] = datos.groupby('place_name')['surface_total_in_m2'].apply(lambda x: x.fillna(x.mean()))
    
    #tamaño cubierto, tomo el total de la fila
    datos["surface_covered_in_m2"] = datos['surface_covered_in_m2'].fillna(datos['surface_total_in_m2'])
  
    
    #relleno el piso y las habitaciones con la moda del barrio
    datos["floor"] = datos.groupby('place_name')['floor'].fillna(datos.dropna(subset=['floor']).floor.value_counts().idxmax())
    datos["rooms"] = datos.groupby('place_name')['rooms'].fillna(datos.dropna(subset=['rooms']).floor.value_counts().idxmax())
    
    
    
     #si en el campo de las expensas dice no, entonces le asigno expensas cero
    datos["expenses"] = datos["expenses"].map(lambda exp: "0" if(re.search(str(exp), 'no', re.IGNORECASE)) else str(exp))
                                                            
    #si no tiene un valor numerico lo cambio a nan para luego asignarle el promedio del barrio                                                   
    datos["expenses"] = datos["expenses"].map(lambda exp: re.sub("[^0-9]", "",str(exp) ))
    datos["expenses"] = datos["expenses"].map(lambda exp: np.NaN if(str(exp)== "") else exp)
    datos["expenses"] = datos["expenses"].map(lambda exp: float(exp))
    datos["expenses"] = datos.groupby('place_name')['expenses'].apply(lambda x: x.fillna(x.mean()))
    
    
    #por si queda algun nan
    datos.fillna(0, inplace=True)
    
    return datos



def delete_signs(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])
def delete_accent_mark(s):
    return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

def unificate_description(d):
    if(pd.notnull(d)):
        return delete_signs(delete_accent_mark(d).lower())
    return d
    



#lista con todos los barrios posibles sin repetir (sin tildes y en minuscula)
def get_neighborhoods(properati_df,nombre_conj_barios):
    neighborhoods = properati_df[pd.notnull(properati_df['place_name'])].place_name
    neighborhoods = neighborhoods.drop_duplicates(keep='first')
    neighborhoods = neighborhoods.map(lambda b: delete_accent_mark(b).lower())
    neighborhoods.replace(nombre_conj_barios, "sin barrio", inplace = True)
    return neighborhoods



def get_place_name(data):

    places =[]
    for index, row in data.iterrows():
        if(pd.notnull(row['place_name_y']) and (row['place_name_y']!="sin barrio") ):
            places.append(row['place_name_y'].title())
        else:
            if(pd.notnull(row['place_name_x'])):
                places.append(row['place_name_x'].title())
            else:
                places.append(row['place_name_x'])
    return places



def assign_neighborhoods(description,neighborhoods):
    for neighborhood in neighborhoods:
        if(str(description.encode('utf-8')).find(neighborhood) >= 0):
              return neighborhood
    return "sin barrio"




def guess_neighborhoods(df,  nombre_conj_barios):
    
    neighborhoods = get_neighborhoods(df,nombre_conj_barios)

    sin_barrio = df.loc[df.place_name.str.contains(nombre_conj_barios , na=False),:]
    sin_barrio = pd.concat([sin_barrio,df.loc[pd.isnull(df.place_name),:]])
    sin_barrio.place_name = sin_barrio.description.map(lambda description: assign_neighborhoods(description,neighborhoods))

    
    barrio_asignado = sin_barrio.loc[sin_barrio.place_name.str.contains("sin barrio", na=False) == False,:]
    

    columns =['country_name', 'created_on', 'currency', 'description',
       'expenses', 'extra', 'floor', 'geonames_id', 'id',
       'image_thumbnail', 'lat', 'lat-lon', 'lon', 'operation',
       'place_with_parent_names', 'price',
       'price_aprox_local_currency', 'price_aprox_usd', 'price_per_m2',
       'price_usd_per_m2', 'properati_url', 'property_type', 'rooms',
       'state_name', 'surface_covered_in_m2', 'surface_in_m2',
       'surface_total_in_m2', 'title','is_to']
    
    
    properati_barrios = pd.merge(df, barrio_asignado,how='outer', on=columns)
    
    properati_barrios['place_name']= get_place_name(properati_barrios)
    columns.append('place_name')
    properati_barrios=properati_barrios.loc[:,columns]
    
    return properati_barrios



def get_df_properati_to_predict(df):

    df.description = df.description.map(lambda d: unificate_description(d))
    df.place_name = df.place_name.map(lambda p: unificate_description(p))
    
    guess_neighborhoods(df,"Capital Federal")
    guess_neighborhoods(df,"Buenos Aires Interior")
    
    return fill_nan_and_convert_to_float(df)



properati_no_price= pd.read_csv('../tp1/properati_dataset_testing_noprice.csv',encoding='UTF-8')
properati_no_price['is_to'] ='predict'

properati_filtered= pd.read_csv('../tp1/properati_filtered1.csv',encoding='UTF-8')
properati_filtered['is_to'] = 'train'

properati_filled =  properati_filtered.append(properati_no_price,ignore_index=True)
properati_filled = get_df_properati_to_predict(properati_filled)



df_to_train= properati_filled.loc[properati_filled['is_to'] == 'train',['created_on','property_type','lat','lon', 'place_name','state_name',
                 'surface_total_in_m2','surface_covered_in_m2','description',
                 'floor','rooms','expenses', 'price_aprox_usd','price_usd_per_m2']]

df_to_predict= properati_filled.loc[properati_filled['is_to'] == 'predict',['id','created_on','property_type','lat','lon', 'place_name','state_name',
                 'surface_total_in_m2','surface_covered_in_m2','description',
                 'floor','rooms','expenses']]