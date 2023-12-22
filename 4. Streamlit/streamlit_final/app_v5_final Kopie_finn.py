import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#### Page Style
#########################################################

# Set Streamlit page configuration with page title and icon
st.set_page_config(
    page_title="The German Car Price Predictor", 
    page_icon="🚗", 
    layout="wide"
    )

logo_path = "/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/projectlogo.jpg"
logo_slide_path = "/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/backg.png"

st.image(logo_path, use_column_width=True)
#### Sidebar
#########################################################

# Display the logo image
st.sidebar.image(logo_slide_path, use_column_width=True)

# Sidebar
st.sidebar.title("Übersicht")
app_mode = st.sidebar.radio(
    "Was willst du machen?",
    ("Start", "Data Exploration", "Prediction", "Modell Performance")
)

#### Load Data and Model
#########################################################

# Define your data loading functions
@st.cache_data
def load_data():
    data = pd.read_csv("Cars_Data_Explorer.csv", sep=";",index_col=0)
    data = data[data["Preis"] < 300000]    
    return data.dropna()

@st.cache_data
def load_data_model():
    cars_model = pd.read_csv("Cars_Data_Model.csv", sep=";",index_col=0) 
    cars_model = cars_model[cars_model["Preis"] < 300000]    
    return cars_model.dropna()


@st.cache_data
def load_scaler():
    filename = "final_scaler.sav"
    loaded_scaler = pickle.load(open(filename, "rb"))
    return loaded_scaler

# Load data and model
data = load_data()
cars = load_data_model()
scaler = load_scaler()

#### Definition of Main Page
#########################################################

# Page Header
def start_page():
    st.title("The German Car Value Predictor")
    st.markdown("Willkommen bei The German Car Value Predictor! Entdecken Sie die Marktwerte von Gebrauchtwagen.")  
    st.header("Über uns")
    st.markdown("The German Car Value Predictor wurde entwickelt, um die Preistransparenz auf dem Gebrauchtwagenmarkt zu erhöhen und damit die Informationsasymmetrie zwischen Experten und Laien zu verringern. Unser Machine-Learning-Modell, das auf einer umfangreichen Datenmenge des Marktführers Autoscout trainiert wurde, ermöglicht eine genaue Preisfindung für verschiedene Fahrzeugmodelle der Marken: Volkswagen, Mercedes-Benz, Audi, BMW, Opel und Porsche. Unsere Modelle werden in regelmässigen Abständen verbessert, um die bestmöglichen Preise zu ermitteln. Die Preise werden auf Basis des deutschen Gebrauchtwagenmarktes ermittelt und sind in Euro (€) angegeben.")
    st.write("")
    st.write("")
    row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])
    with row1_col1:
        st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/vw.png", use_column_width=True)
        st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/bmw.png", use_column_width=True)
        
    with row1_col2:
        st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/opel.png", use_column_width=True)
        st.write("")
        st.write("")
        st.write("")
        st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/benz.png", use_column_width=True)
    with row1_col3:
        st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/porsche.png", use_column_width=True)
        st.write("")
        st.write("")
        st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/audi.png", use_column_width=True)

    st.write("")
    st.write("")
    st.subheader("Für wen eignet sich The German Car Value Predictor?")
    st.markdown("Von der einfachen Preisschätzung des Marktwertes des eigenen Autos durch den normalen Autofahrer, über die Ermittlung ganzer Preislisten durch Autohändler, bis hin zur Modellauswertung durch den Datenenthusiasten, bietet The German Car Value Predictor für jeden etwas. Ziel ist es, nicht nur exakte Preise zu ermitteln, sondern auch durch die Bereitstellung des Datensatzes und verschiedener Kennzahlen zu den Modellen Transparenz zu schaffen, um Informationsasymmetrien im Markt zu bekämpfen.")
    st.subheader("Viel Spass beim Ausprobieren ! ")
    st.write("")
    
  
        
            
#### Definition Data Exploration
#########################################################

def data_exploration():
    # Title 
    st.title("Data Exploration")
    # Header
    st.header("Filter")
    # Colors
    palette_red = ["#8B0000","#CA907E","#BA274A","#841C26", "#E88D67","#201E50","#525B76", "#C96480","#987284","#F9B5AC", "#0A1045","#8D5A97", "#EF476F", "#FFD166", "#E88873", "#E0DDCF"]
    
    #### Filtering
    
    # Introducing 5 colums for user inputs
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns([1,1,1,1,1])
    
    row2_col1 = st.columns([1])[0]
    
    price = row1_col2.slider("Preis des Autos",
                      data["Preis"].min(),
                      data["Preis"].max(),
                      (100.0, 1990000.0))
    
    kilometerstand = row1_col3.slider("Kilometerstand des Autos",
                      data["Kilometerstand"].min(),
                      data["Kilometerstand"].max(),
                      (0, 999999))
    
    ps = row1_col4.slider("PS des Autos",
                      data["PS"].min(),
                      data["PS"].max(),
                      (1, 999))
    
    # Get unique values from the 'Marke' column
    unique_brand = data["Marke"].unique()
    options = ["Alle Marken"] + list(unique_brand)
    selected_brand = row1_col1.selectbox("Wähle deine Automarke", options, key="selected_brand")
    
    # After selecting a brand, filter models based on the selected brand
    if selected_brand != "Alle Marken":
        models_in_brand = data[data["Marke"] == selected_brand]["Modell"].unique()
        model_options = ["Alle Modelle"] + list(models_in_brand)
    else:
        model_options = ["Alle Modelle"] + list(data["Modell"].unique())
    
    selected_model = row2_col1.selectbox("Wähle dein Modell", model_options, key="selected_model")
    
    if selected_brand == "Alle Marken" :
        if selected_model == "Alle Modelle" :
            filtered_data = data.loc[(data["Preis"] >= price[0]) &
                                     (data["Preis"] <= price[1]) &
                                     (data["Kilometerstand"] >= kilometerstand[0]) & 
                                     (data["Kilometerstand"] <= kilometerstand[1]) &
                                     (data["PS"] >= ps[0]) & 
                                     (data["PS"] <= ps[1]), :]
        else:
            # creating filtered data set according to slider inputs
            filtered_data = data.loc[(data["Modell"] == selected_model) &
                                     (data["Preis"] >= price[0]) &
                                     (data["Preis"] <= price[1]) &
                                     (data["Kilometerstand"] >= kilometerstand[0]) & 
                                     (data["Kilometerstand"] <= kilometerstand[1]) &
                                     (data["PS"] >= ps[0]) & 
                                     (data["PS"] <= ps[1]), :]
          
    else:
        if selected_model == "Alle Modelle" :
            filtered_data = data.loc[(data["Marke"] == selected_brand) &
                                     (data["Preis"] >= price[0]) &
                                     (data["Preis"] <= price[1]) &
                                     (data["Kilometerstand"] >= kilometerstand[0]) & 
                                     (data["Kilometerstand"] <= kilometerstand[1]) &
                                     (data["PS"] >= ps[0]) & 
                                     (data["PS"] <= ps[1]), :]
        else:
        # creating filtered data set according to slider inputs
            filtered_data = data.loc[(data["Marke"] == selected_brand) &
                                     (data["Modell"] == selected_model) &
                                     (data["Preis"] >= price[0]) &
                                     (data["Preis"] <= price[1]) &
                                     (data["Kilometerstand"] >= kilometerstand[0]) & 
                                     (data["Kilometerstand"] <= kilometerstand[1]) &
                                     (data["PS"] >= ps[0]) & 
                                     (data["PS"] <= ps[1]), :]

    # Calculate Average Prices
    average_price_karosserie = filtered_data.groupby('Karosserieform')['Preis'].mean().reset_index()
    average_price_innenausstattung = filtered_data.groupby('Innenausstattung')['Preis'].mean().reset_index()
    average_price_getriebe = filtered_data.groupby('Getriebe')['Preis'].mean().reset_index()
    average_price_aussenfarbe = filtered_data.groupby('Außenfarbe')['Preis'].mean().reset_index()
    average_price_media = filtered_data.groupby('Unterhaltung/Media')['Preis'].mean().reset_index()
    
    #### Show Filtered Data
    
    # Add checkbox allowing us to display raw data
    if row1_col5.toggle("Daten Anzeigen", False):
        st.subheader("Gefilterte Autos")
        # Reset the index and drop the old one
       # Set a column as the index of the DataFrame
        filtered_data = filtered_data.set_index('Marke', drop=True)

        st.dataframe(filtered_data)
        
    if row1_col5.toggle("Grafiken", False):
        st.subheader("Statistiken zu den ausgewählten Filter")
        standard_fig_size = (7, 6)
        
        # creating tabs
        tab1, tab2, tab3 = st.tabs(["Verteilung", "Technologische Faktoren", "Design und Ausstattung"])
        
        with tab1:
            # defining two columns for layouting plots 
            row3_col1, row3_col2, row3_col3  = st. columns([1,1,1])
            
            # Histogram of Prices in the first column  
            with row3_col1:
                st.subheader('Preis')
                fig, ax = plt.subplots(figsize = standard_fig_size, facecolor='#A9BFC7')
                sns.histplot(filtered_data['Preis'], bins=50, kde=False, ax=ax, color='#8B0000')
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Preis', fontsize = 20)
                ax.set_ylabel('Anzahl', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
                
            with row3_col2:
                st.subheader('Kilometerstand')
                fig, ax = plt.subplots(figsize = standard_fig_size, facecolor='#A9BFC7')
                sns.histplot(filtered_data['Kilometerstand'], bins=50, kde=False, ax=ax, color='#8B0000')
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Kilometerstand', fontsize = 20)
                ax.set_ylabel('Anzahl', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
                    
            with row3_col3:
                st.subheader('Erstzulassung')
                fig, ax = plt.subplots(figsize = standard_fig_size, facecolor='#A9BFC7')
                sns.histplot(filtered_data['Erstzulassung'], bins=50, kde=False, ax=ax, color='#8B0000')
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Jahr', fontsize = 20)
                ax.set_ylabel('Anzahl', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
         
        
        with tab2:
            # defining two columns for layouting plots 
            row4_col1, row4_col2, row4_col3  = st. columns([1,1,1])
    
            with row4_col1:
                st.subheader('Getriebe') 
                fig, ax = plt.subplots(figsize = standard_fig_size, facecolor='#A9BFC7')
                sns.barplot(x='Getriebe', y='Preis', data=average_price_getriebe, palette=palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Getriebe', fontsize = 20)
                ax.set_ylabel('Durchschnittspreis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
                
            # Boxplot of Kilometerstand by Brand in the second column
            with row4_col2:
                st.subheader('Pferdestärke PS')
                fig, ax = plt.subplots(figsize = standard_fig_size, facecolor='#A9BFC7')
                sns.scatterplot(x='PS', y='Preis', data=filtered_data, hue='Marke', ax=ax, palette = palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('PS', fontsize = 20)
                ax.set_ylabel('Preis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
                
            # Scatter Plot of Preis vs. Kilometerstand in the third column
            with row4_col3:
                st.subheader('Kilometerstand')
                fig, ax = plt.subplots(figsize = standard_fig_size, facecolor='#A9BFC7')
                sns.scatterplot(x='Kilometerstand', y='Preis', data=filtered_data, hue='Marke', ax=ax, palette = palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Kilometerstand', fontsize = 20)
                ax.set_ylabel('Preis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')     
                   
        with tab3:
            # defining two columns for layouting plots 
            row5_col1, row5_col2 = st. columns([1,1])
            row6_col1, row6_col2 = st. columns([1,1])
            
            with row5_col1:
                st.subheader("Karosserieform")
                   
                fig, ax = plt.subplots(figsize = (16, 6), facecolor='#A9BFC7')
                sns.barplot(x='Karosserieform', y='Preis', data=average_price_karosserie, palette=palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Karosserieform', fontsize = 20)
                ax.set_ylabel('Durchschnittspreis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
               
            with row5_col2:
                st.subheader('Innenausstattung')
                
                fig, ax = plt.subplots(figsize = (16, 6), facecolor='#A9BFC7')
                sns.barplot(x='Innenausstattung', y='Preis', data=average_price_innenausstattung, palette=palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Innenausstattung', fontsize = 20)
                ax.set_ylabel('Durchschnittspreis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
            
            with row6_col1:
                st.subheader('Aussenfarbe')
                
                fig, ax = plt.subplots(figsize = (16, 6), facecolor='#A9BFC7')
                sns.barplot(x='Außenfarbe', y='Preis', data=average_price_aussenfarbe, palette=palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Ausssenfarbe', fontsize = 20)
                ax.set_ylabel('Durchschnittspreis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')
            
            with row6_col2:
                st.subheader('Unterhaltung und Media')
                
                fig, ax = plt.subplots(figsize = (16, 6), facecolor='#A9BFC7')
                sns.barplot(x='Unterhaltung/Media', y='Preis', data=average_price_media, palette=palette_red)
                ax.set_facecolor('#A9BFC7')
                ax.set_xlabel('Unterhaltung/Media', fontsize = 20)
                ax.set_ylabel('Durchschnittspreis', fontsize = 20)
                for spine in ax.spines.values():
                   spine.set_edgecolor('#26404D')
                st.pyplot(fig, bbox_inches='tight')


#### Static Plots 
#########################################################

def static_plots():   
    st.header("Statistiken zum gesamten Dataset")
    
    row6_col1, row6_col2 = st. columns([1,1])
    palette_red = ["#8B0000","#CA907E","#BA274A","#841C26", "#E88D67","#201E50","#525B76", "#C96480","#987284","#F9B5AC", "#0A1045","#8D5A97", "#EF476F", "#FFD166", "#E88873", "#E0DDCF"]
    fixed_radius=0.9
    standard_fig_size_pie = (14, 14)
    
    with row6_col1:
        st.subheader('Verteilung Marken')
        fig, ax = plt.subplots(figsize=standard_fig_size_pie, facecolor='#A9BFC7')
        data['Marke'].value_counts().plot(kind='pie', 
                                          autopct='%1.1f%%', 
                                          colors=palette_red, 
                                          radius=fixed_radius,
                                          legend=True)
        ax.set_facecolor('#A9BFC7')
        ax.set_ylabel('')
        plt.setp(ax.texts, text="")
        ax.legend(loc='lower center', 
                  bbox_to_anchor=(0.5, -0.1), 
                  ncol=3, frameon=False, 
                  fontsize=20)
        for spine in ax.spines.values():
            spine.set_edgecolor('#26404D')
        st.pyplot(fig, bbox_inches='tight')
    
    with row6_col2:   
        # Function to calculate statistics and return as a dictionary
        def calculate_statistics(df):
            stats = {
                'Total Count': len(df),
                'Brand Distribution': df['Marke'].value_counts().to_dict(),
                'Brand Percentages': (df['Marke'].value_counts(normalize=True) * 100).round(2).to_dict(),
                'Average Price': round(df['Preis'].mean(), 2),
                'Price Std Deviation': round(df['Preis'].std(), 2),
                'Average Price by Brand': df.groupby('Marke')['Preis'].mean().round(2).to_dict(),
                'Std Dev Price by Brand': df.groupby('Marke')['Preis'].std().round(2).to_dict(),
                'Average Kilometerstand': round(df['Kilometerstand'].mean(), 2),
                'Kilometerstand Std Deviation': round(df['Kilometerstand'].std(), 2),
                'Average PS': round(df['PS'].mean(), 2),
                'PS Std Deviation': round(df['PS'].std(), 2),
                'Average KW': round(df['KW'].mean(), 2),
                'KW Std Deviation': round(df['KW'].std(), 2),
                'Aussenfarbe Distribution': df['Außenfarbe'].value_counts().to_dict()
            }
            return stats
        
        # Calculate statistics
        with st.container():
            car_stats = calculate_statistics(data)
            st.metric("Anzahl Einträge",value=car_stats.get('Total Count'))
            row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])
            row2_col1, row2_col2, row2_col3 = st.columns([1,1,1])
          
             
            with row1_col1:
                st.metric("⌀ Preis",value=car_stats.get('Average Price'))
            with row1_col2:
                st.metric("⌀ Kilometerstand",value=car_stats.get('Average Kilometerstand'))
            with row1_col3:
                st.metric("⌀ PS",value=car_stats.get('Average PS'))
            
            with row2_col1:
                st.metric("Preis Std.",value=car_stats.get('Price Std Deviation'))
            with row2_col2:               
                st.metric("Kilometerstand Std.",value=car_stats.get('Kilometerstand Std Deviation'))
            with row2_col3:
                st.metric("PS Std.",value=car_stats.get('PS Std Deviation'))
            
            st.write()
            
            # Convert to DataFrame
            df_car_stats = pd.DataFrame.from_dict(car_stats)
            # List of row labels to be dropped
            columns_to_drop = ["Total Count", "Average Price", "Price Std Deviation", 
                            "Average Kilometerstand", "Kilometerstand Std Deviation", 
                            "Average PS", "PS Std Deviation", "Average KW", 
                            "KW Std Deviation", "Aussenfarbe Distribution"]
            rows_to_drop = ["Schwarz", "Grau", "Weiß", "Silber", "Blau",
                           "Rot", "Grün", "Braun", "Beige", "Orange",
                           "Gelb", "Violett", "Gold", "Bronze"]
           
            df_car_stats = df_car_stats.rename(columns={
                'Brand Distribution': 'Anzahl Einträge',
                'Brand Percentages': '% Anteil der Marke',
                'Average Price by Brand': '⌀ Preis',
                'Std Dev Price by Brand': 'Std. Preis',
})

            # Drop the specified rows
            df_car_stats = df_car_stats.drop(columns_to_drop, axis=1)
            df_car_stats = df_car_stats.drop(rows_to_drop, axis=0)
            
            st.dataframe(df_car_stats)

#### Definition of Section Prediction
#########################################################

def input_df_to_model_input_df(input_df):
    
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    
    cars_model = load_data_model()
  
    #Label Encoding
  
    # Feature Komfort
    label_encoder_komfort = LabelEncoder()
    classes_komfort = ["Hoch", "Mittel", 'Niedrig']
    label_encoder_komfort.fit(classes_komfort)
    # Wende Label Encoding auf die Spalte "Komfort" in input_df an
    input_df["Komfort"] = label_encoder_komfort.transform(input_df["Komfort"])
  
    # Feature Sicherheit
    label_encoder_sicherheit = LabelEncoder()
    sicherheit_classes = ["Hoch", "Mittel", "Niedrig"]
    label_encoder_sicherheit.fit(sicherheit_classes)
    input_df["Sicherheit"] = label_encoder_sicherheit.transform(input_df["Sicherheit"])
  
    # Feature Extras
    label_encoder_extras = LabelEncoder()
    extras_classes = ["Hoch", "Mittel", "Niedrig"]
    label_encoder_extras.fit(extras_classes)
    input_df["Extras"] = label_encoder_extras.transform(input_df["Extras"])
  
    # Feature Unterhaltung/Media
    label_encoder_unterhaltung = LabelEncoder()
    unterhaltung_classes = ["Hoch", "Mittel", "Niedrig"]
    label_encoder_unterhaltung.fit(unterhaltung_classes)
    input_df["Unterhaltung/Media"] = label_encoder_unterhaltung.transform(input_df["Unterhaltung/Media"])
  
    # Feature Fahrzeugzustand
    label_encoder_fahrzeugzustand = LabelEncoder()
    fahrzeugzustand_classes = ['fahrtauglich, nicht repariert, kein Unfallauto', 'Repariert', 'UnfallfahrzeugRepariert', 'Unfallfahrzeug',
                             'Nicht fahrtauglich', 'RepariertNicht fahrtauglich', 'UnfallfahrzeugNicht fahrtauglich']
    label_encoder_fahrzeugzustand.fit(fahrzeugzustand_classes)
    input_df["Fahrzeugzustand"] = label_encoder_fahrzeugzustand.transform(input_df["Fahrzeugzustand"])
  
    # Feature Schadstoffklasse
    label_encoder_schadstoffklasse = LabelEncoder()
    schadstoffklasse_classes = ['Keine', 'Euro 1', 'Euro 2', 'Euro 3', 'Euro 4', 'Euro 5', 'Euro 6', 'Euro 6c', 'Euro 6d', 'Euro 6d-TEMP']
    label_encoder_schadstoffklasse.fit(schadstoffklasse_classes)
    input_df["Schadstoffklasse"] = label_encoder_schadstoffklasse.transform(input_df["Schadstoffklasse"])
  
    #One-Hot Encoding
  
    # Liste der Spalten für One-Hot-Encoding
    columns_to_encode = ['Marke', 'Modell', 'Fahrzeugart', 'Karosserieform', 'Getriebe',
                        'Außenfarbe', 'Innenausstattung', 'Nichtraucherfahrzeug',
                        'Scheckheftgepflegt', 'Garantie','Farbe der Innenausstattung', 'Taxi oder Mietwagen']
  
    # Führe One-Hot-Encoding für die ausgewählten Spalten durch
    input_df = pd.get_dummies(input_df, columns=columns_to_encode, prefix=columns_to_encode)
  
    # Vergleiche Spalten und füge fehlende Spalten in input_df ein
    missing_columns = list(set(cars_model.columns) - set(input_df.columns))
    if missing_columns:
      missing_df = pd.DataFrame(0, columns=missing_columns, index=input_df.index)
      input_df = pd.concat([input_df, missing_df], axis=1)
  
  
    # Fülle NaN-Werte in input_df mit 0
    input_df = input_df.fillna(0)
  
    # Preis Spalte droppen
    input_df = input_df.drop("Preis", axis=1)
  
    #Fall NaN Werte mit 0 füllen
    input_df = input_df.fillna(0)
    
    input_df = input_df[cars_model.columns.drop("Preis")]
    
    # Return-Statement
    return input_df

###
def predicting_uploaded_data():
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    import numpy as np


  
    input_customer = st.file_uploader("Lade das Template File hoch, um deine Preise zu entdecken.")
    
    if input_customer is not None:

        input_df = pd.read_excel(input_customer,index_col=0)
        output_df = pd.read_excel(input_customer)
       
        output_df = output_df.dropna()
        input_df = input_df.dropna()
        
        processed_df = input_df_to_model_input_df(input_df)
       
        with st.spinner('Unser Model berechnet die Preis Vorhersage...'):
            
            # Aufteilung der Daten in Marken-DataFrames
            audi = processed_df[processed_df["Marke_audi"] == 1]
            bmw = processed_df[processed_df["Marke_bmw"] == 1]
            mercedes = processed_df[processed_df["Marke_mercedes-benz"] == 1]
            opel = processed_df[processed_df["Marke_opel"] == 1]
            porsche = processed_df[processed_df["Marke_porsche"] == 1]
            volkswagen = processed_df[processed_df["Marke_volkswagen"] == 1]
        
            # Eine Liste der Marken und ihrer entsprechenden DataFrames erstellen
            brand_dataframes = {
                'audi': audi,
                'bmw': bmw,
                'mercedes-benz': mercedes,
                'opel': opel,
                'porsche': porsche,
                'volkswagen': volkswagen
            }
                    # Liste zur Speicherung der skalierten DataFrames
            scaled_dataframes = []
            
            for brand, df in brand_dataframes.items():
                if len(df) > 0:
                    # Skalieren des DataFrames für die aktuelle Marke
                    scaled_df = scaler.transform(df)
                    
                    # Konvertieren des skalierten numpy Arrays zurück in einen DataFrame
                    scaled_df = pd.DataFrame(scaled_df, columns=df.columns, index=df.index)
                    
                    # Hinzufügen des skalierten DataFrames zur Liste
                    scaled_dataframes.append(scaled_df)
            
            # Zusammenführen aller skalierten DataFrames in einen Gesamtdatenrahmen
            total_scaled_df = pd.concat(scaled_dataframes)  
            st.write(total_scaled_df)
            st.write(scaled_dataframes)

         
            # for brand, df in brand_dataframes.items():
            #     if len(df) > 0:
            #         scaled_data = scaler.transform(df)
            # scaled_data = pd.DataFrame(scaled_data, columns=processed_df.columns, index=processed_df.index)
                    
            st.write(total_scaled_df)
            
            
            # # SCALER
            # scaled_data = scaler.transform(processed_df)
            #predicting
            output_df["predicted_price"] = np.exp(model.predict(total_scaled_df))

            
            cols = list(output_df.columns)
            cols.insert(3, cols.pop(-1))
            output_df = output_df[cols]
            st.write(output_df)
            
        # Add User Feedback
        st.success("Der Preis von %i Auto wurde bestimmt! Ihre Excel-Datei steht für dich zum Download bereit." % output_df.shape[0])
        
    # Add Download Button
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            output_df.to_excel(writer, index=False)
                # Note: Removed writer.save() as it's not needed
    
        output.seek(0)
    
        st.download_button(
            label="Download File",
            data=output,
            file_name="CARVALUEPREDICTOR_scored_car_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
            
        
###
def predicting_one_car(): 
    import numpy as np
    import pandas as pd
 
    
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns([1,1,1,1,1])
    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns([1,1,1,1,1])
    row3_col1, row3_col2, row3_col3, row3_col4, row3_col5 = st.columns([1,1,1,1,1])
    row4_col1, row4_col2, row4_col3, row4_col4, row4_col5 = st.columns([1,1,1,1,1])
    row5_col1, row5_col2, row5_col3, row5_col4, row5_col5 = st.columns([1,1,1,1,1])
    

 
    brand_option= data["Marke"].unique()
    model_option = data["Modell"].unique()
    
    fahrzeugart_option = data["Fahrzeugart"].unique()
    karosserieform_option = data["Karosserieform"].unique()
    getriebe_option = data["Getriebe"].unique()
    tueren_option = data["Türen"].unique() 
    aussenfarbe_option = data["Außenfarbe"].unique()
    komfort_option = data["Komfort"].unique()
    sicherheit_option = data["Sicherheit"].unique()
    extras_option = data["Extras"].unique()
    media_option = data["Unterhaltung/Media"].unique()
    innenausstattung_option = data["Innenausstattung"].unique()
    farbeinnen_option = data["Farbe der Innenausstattung"].unique()
    schadstoffklasse_option = data["Schadstoffklasse"].unique()
    rauchen_option = data["Nichtraucherfahrzeug"].unique()
    scheckheft_option = data["Scheckheftgepflegt"].unique()
    garantie_option = data["Garantie"].unique()
    fahrzeugzustand_option = data["Fahrzeugzustand"].unique()
    taxi_option = data["Taxi oder Mietwagen"].unique()
    innenausstattung_option = data["Innenausstattung"].unique()

    # variabel          
    var_brand = row1_col1.selectbox("Marke", brand_option, key="var_brand")
    
    if var_brand:
        models_in_brand = data[data["Marke"] == var_brand]["Modell"].unique()
        model_option = ["Alle Modelle"] + list(models_in_brand)
    
    var_model = row1_col2.selectbox("Modell", model_option, key="var_model")
    var_fahrzeug = row1_col3.selectbox("Fahrzeugart", fahrzeugart_option, key="var_fahrzeug")
    var_karosserie = row1_col4.selectbox("Karosserieform", karosserieform_option, key="var_karosserie" )
    var_kilometer = row1_col5.number_input("Kilometerstand", value=None, placeholder="Zahl", key="var_kilometer ")
        
    var_fahrleistung = row2_col1.number_input("Fahrleistung p.a.", value=None, placeholder="Zahl", key = "var_fahrleistung")
    var_getriebe = row2_col2.selectbox("Getriebe", getriebe_option, key="var_getriebe")
    var_tueren = row2_col3.selectbox("Türen", tueren_option, key="var_tueren")
    var_aussenfarbe = row2_col4.selectbox("Aussenfarbe", aussenfarbe_option, key="var_aussenfarbe")
    var_komfort = row2_col5.selectbox("Komfort", komfort_option, key="var_komfort")
    
    var_sicherheit = row3_col1.selectbox("Sicherheit", sicherheit_option, key="var_sicherheit")  
    var_extras = row3_col2.selectbox("Extras", extras_option, key="var_extras")
    var_erstzulassung = row3_col3.number_input("Erstzulassung", value=None, placeholder="Zahl", key= "var_erstzulassung")
    var_media = row3_col4.selectbox("Media", media_option, key="var_media")
    var_hubraum = row3_col5.number_input("Hubraum", value=None, placeholder="Zahl", key= "var_hubraum")
    
    var_innenausstattung = row4_col1.selectbox("Innenausstattung", innenausstattung_option, key="var_innenausstattung")
    var_farbeinnen = row4_col2.selectbox("Farbe der Innenausstattung", farbeinnen_option, key="var_farbeinnen")
    var_schadstoffklasse = row4_col3.selectbox("Schadstoffklasse", schadstoffklasse_option, key="var_schadstoffklasse")
    var_rauchen = row4_col4.selectbox("Nichtraucherfahrzeug", rauchen_option, key="var_rauchen")
    var_scheckheft = row4_col5.selectbox("Scheckheftgepflegt", scheckheft_option, key="var_scheckheft")
    
    var_garantie = row5_col1.selectbox("Garantie", garantie_option, key="var_garantie")
    var_farhzeugzustand = row5_col2.selectbox("Fahrzeugzustand", fahrzeugzustand_option, key="var_farhzeugzustand")
    var_taxi = row5_col3.selectbox("Taxi oder Mietwagen", taxi_option, key="var_taxi")
    var_kw = row5_col4.number_input("KW", value=None, placeholder="Zahl", key = "var_kw")
    var_ps = row5_col5.number_input("PS", value=None, placeholder="Zahl", key= "var_ps")

    # create dataframe 

# Creating a DataFrame with these variables
    input_einzel_df = pd.DataFrame({
        'Marke': [var_brand],
        'Modell': [var_model],
        'Fahrzeugart': [var_fahrzeug],
        'Karosserieform': [var_karosserie],
        'Kilometerstand': [var_kilometer],
        'Fahrleistung p.a.': [var_fahrleistung],
        'Getriebe': [var_getriebe],
        'Türen': [var_tueren],
        'Außenfarbe': [var_aussenfarbe],
        'Komfort': [var_komfort],
        'Sicherheit': [var_sicherheit],
        'Extras': [var_extras],
        'Erstzulassung': [var_erstzulassung],
        'Unterhaltung/Media': [var_media],
        'Hubraum': [var_hubraum],
        'Innenausstattung': [var_innenausstattung],
        'Farbe der Innenausstattung': [var_farbeinnen],
        'Schadstoffklasse': [var_schadstoffklasse],
        'Nichtraucherfahrzeug': [var_rauchen],
        'Scheckheftgepflegt': [var_scheckheft],
        'Garantie': [var_garantie],
        'Fahrzeugzustand': [var_farhzeugzustand],
        'Taxi oder Mietwagen': [var_taxi],
        'KW': [var_kw],
        'PS': [var_ps]
    })

    
    if st.button("Predict", type="primary", use_container_width=True):
        import pandas as pd
        


        
        with st.spinner('Model is predicting. Bitte warte...'):
            processed_einzel_df = input_df_to_model_input_df(input_einzel_df)
            
            # Aufteilung der Daten in Marken-DataFrames
            audi = processed_einzel_df[processed_einzel_df["Marke_audi"] == 1]
            bmw = processed_einzel_df[processed_einzel_df["Marke_bmw"] == 1]
            mercedes = processed_einzel_df[processed_einzel_df["Marke_mercedes-benz"] == 1]
            opel = processed_einzel_df[processed_einzel_df["Marke_opel"] == 1]
            porsche = processed_einzel_df[processed_einzel_df["Marke_porsche"] == 1]
            volkswagen = processed_einzel_df[processed_einzel_df["Marke_volkswagen"] == 1]
        
            # Eine Liste der Marken und ihrer entsprechenden DataFrames erstellen
            brand_dataframes = {
                'audi': audi,
                'bmw': bmw,
                'mercedes-benz': mercedes,
                'opel': opel,
                'porsche': porsche,
                'volkswagen': volkswagen
            }
                               
            for brand, df in brand_dataframes.items():
                if len(df) > 0:
                    scaled_einzel_data = scaler.transform(df)
            scaled_einzel_data = pd.DataFrame(scaled_einzel_data, columns=processed_einzel_df.columns, index=processed_einzel_df.index)
                    
            st.write(scaled_einzel_data)
            
            # #scaler
            # scaled_einzel_data = scaler.transform(brand_dataframes)
           
            #predicting
            input_einzel_df["predicted_price"] = np.exp(model.predict(scaled_einzel_data))
        
            
        einzel_preis = input_einzel_df["predicted_price"]
        
        st.success("Der Preis von %i Auto wurde bestimmt! Ihre Excel-Datei steht für dich zum Download bereit." % processed_einzel_df.shape[0])
        st.subheader("Dein Preis") 
        st.write("Der Preis deines Autos beträgt:", einzel_preis)
        
        cols = list(input_einzel_df.columns)
        cols.insert(3, cols.pop(-1))
        input_einzel_df = input_einzel_df[cols]  
       
        import io
        output = io.BytesIO()
    
        # Use ExcelWriter to write to the BytesIO object
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            input_einzel_df.to_excel(writer, index=False)
            # No need for writer.save() here
    
        # Go back to the start of the BytesIO object
        output.seek(0)
    
        # Add Download Button
        st.download_button(
            label="Download File",
            data=output,
            file_name="CARVALUEPREDICTOR_scored_car.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

###
def data_prediction():

    template = pd.read_csv("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/Excel_Upload_Template.csv")
    st.title("Prediction")
    
    st.header("Bestimme den Preis eines Autos")    
    
    predicting_one_car()
    
    st.header("Bestimme den Preis mehrerer Autos")   
    st.markdown("Bitte nutze das Teamplate zur Bestimmung mehrerer Preise. Lade das Template mit den eingetragenen Werten hoch.")
    
    template = open("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/Excel_Upload_Template.csv")
    st.download_button(label="Download Template",
                        data = template,
                        file_name='prediction_template.csv',
                        mime="text/csv"
                        )
    
    predicting_uploaded_data()
    

#### Definition of Section Modell Performance
#########################################################
    
def model_info():
    st.title("Modell Performance")    
    st.header("Aktuelles Modell")
    st.markdown("blablabla")
    
    st.subheader("Performance")
    st.markdown("blablabla")
    
    st.header("Modelling Prozess")
    st.subheader("Research Question")
    st.markdown("Wie kann ein in einer benutzerfreundlichen Web-App integriertes Machine Learning-Modell sowohl Verkaufenden als auch Kaufenden ermöglichen, effizient und effektiv Informationsasymmetrien zu verringern und dadurch zu einer fairen und transparenten Preisfindung auf dem deutschen Gebrauchtwagenmarkt beitragen?")
    st.subheader("Übersicht")
    st.markdown("In der folgenden Übersicht sind verschiedene Modelle, die während des Modellierungsprozesses getestet wurden, sowie ihre Metriken und Hyperparameter aufgeführt.")
    st.image("/Users/DarioPatzi/Desktop/Business Analytics/Module 05 BAD/final_app_debugged/modelling.png", use_column_width=True)

#### Navigation
#########################################################
if app_mode == "Start": 
   start_page()

if app_mode == "Data Exploration":
    data_exploration()
    static_plots()


if app_mode == "Prediction":
    
    from sklearn.ensemble import GradientBoostingRegressor
    
    class KomboGradientBoostingLog:
        def __init__(self):
            self.models = {
                'audi': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1),
                'volkswagen': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.2),
                'porsche': GradientBoostingRegressor(n_estimators=100, max_depth=7, learning_rate=0.2),
                'opel': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1),
                'mercedes-benz': GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.1),
                'bmw': GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.1)
            }
    
        def fit(self, brand_datasets):
            for brand, datasets in brand_datasets.items():
                if brand in self.models:
                    self.models[brand].fit(datasets['X_train'], datasets['y_train'])
    
        def predict(self, X):
            predictions = pd.Series(index=X.index)
    
            # Verarbeiten Sie jede Marke einzeln und überprüfen Sie, ob Daten vorhanden sind
            for brand in ['audi', 'volkswagen', 'bmw', 'mercedes-benz', 'opel', 'porsche']:
                brand_column = f'Marke_{brand}'
                if brand_column in X.columns and not X[X[brand_column] == 1].empty:
                    predictions[X[brand_column] == 1] = self.models[brand].predict(X[X[brand_column] == 1])
    
            return predictions







    @st.cache_data
    def load_model():
        filename = "final_KomboGradientBoostingLog_model.sav"
        loaded_model = pickle.load(open(filename, "rb"))
        return loaded_model
    model=load_model()
    
    data_prediction()
    
if app_mode == "Modell Performance":
    model_info()




















