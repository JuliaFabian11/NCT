# =============================================================================
# PROYECTO: Expansión Infinita o Desigualdad Masiva - Análisis de Exposición
# de NCT (Neo Culture Technology)
# Autor: Julia Fabian Viamonte - Pensamiento Computacional para Comunicadores
# =============================================================================
# IMPORTAMOS LAS BIBLIOTECAS NECESARIAS
# streamlit para construir la interfaz web
import streamlit as st
# streamlit_option_menu para crear el menú horizontal de navegación
from streamlit_option_menu import option_menu
# pandas para leer y manipular los archivos csv
import pandas as pd
# numpy para operaciones numéricas auxiliares (promedios, redondeos, arrays)
import numpy as np
# plotly.express y plotly.graph_objects para los gráficos interactivos
import plotly.express as px
import plotly.graph_objects as go
# folium y streamlit_folium para el mapa interactivo de origen de integrantes
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
# random para el juego "Adivina el integrante"
import random
# datetime para el manejo de fechas y la conversión de duraciones de video
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# =============================================================================
# Configuramos el título de la pestaña, el ícono y el layout ancho para
# aprovechar todo el espacio horizontal disponible
st.set_page_config(
    page_title="NCT Data Hub",
    page_icon="🎤",
    layout="wide"
)

# =============================================================================
# DICCIONARIO DE APOYO: CIUDADES DE NACIMIENTO -> COORDENADAS Y PAÍS
# =============================================================================
# El archivo miembros_nct.csv solo trae la ciudad de nacimiento (columna
# "LUGAR DE NACIMIENTO"), por lo que para poder ubicar a cada integrante en
# el mapa mundial necesitamos asociar cada ciudad con su país y sus
# coordenadas geográficas (latitud, longitud). Este diccionario cubre
# exactamente las ciudades presentes en el csv de miembros.
CIUDADES_INFO = {
    "Gwanak-gu":  {"pais": "Corea del Sur", "lat": 37.4784, "lon": 126.9516},
    "Seúl":       {"pais": "Corea del Sur", "lat": 37.5665, "lon": 126.9780},
    "Chicago":    {"pais": "Estados Unidos", "lat": 41.8781, "lon": -87.6298},
    "Kadoma":     {"pais": "Japón", "lat": 34.7490, "lon": 135.5850},
    "Guri":       {"pais": "Corea del Sur", "lat": 37.5943, "lon": 127.1296},
    "Vancouver":  {"pais": "Canadá", "lat": 49.2827, "lon": -123.1207},
    "Gimpo":      {"pais": "Corea del Sur", "lat": 37.6152, "lon": 126.7159},
    "Jilin":      {"pais": "China", "lat": 43.8378, "lon": 126.5495},
    "Incheon":    {"pais": "Corea del Sur", "lat": 37.4563, "lon": 126.7052},
    "Busan":      {"pais": "Corea del Sur", "lat": 35.1796, "lon": 129.0756},
    "Shanghai":   {"pais": "China", "lat": 31.2304, "lon": 121.4737},
    "Fújiàn":     {"pais": "China", "lat": 26.0745, "lon": 119.2965},
    "Bangkok":    {"pais": "Tailandia", "lat": 13.7563, "lon": 100.5018},
    "Zhejiang":   {"pais": "China", "lat": 30.2741, "lon": 120.1551},
    "Guangdong":  {"pais": "China", "lat": 23.1291, "lon": 113.2644},
    "Macao":      {"pais": "China", "lat": 22.1987, "lon": 113.5439},
    "Taiwán":     {"pais": "Taiwán", "lat": 25.0330, "lon": 121.5654},
    "Mokpo":      {"pais": "Corea del Sur", "lat": 34.8118, "lon": 126.3922},
    "Fukui":      {"pais": "Japón", "lat": 36.0652, "lon": 136.2216},
    "Tokio":      {"pais": "Japón", "lat": 35.6762, "lon": 139.6503},
    "Daegu":      {"pais": "Corea del Sur", "lat": 35.8714, "lon": 128.6014},
    "Kioto":      {"pais": "Japón", "lat": 35.0116, "lon": 135.7681},
    "Saitama":    {"pais": "Japón", "lat": 35.8617, "lon": 139.6455},
}

# Función que devuelve la información geográfica de una ciudad, con un
# valor por defecto en caso de que la ciudad no esté en el diccionario
def obtener_info_ciudad(ciudad):
    return CIUDADES_INFO.get(ciudad, {"pais": "Desconocido", "lat": 0.0, "lon": 0.0})

# =============================================================================
# CARGA Y LIMPIEZA DE DATOS (con cache para no recalcular en cada interacción)
# =============================================================================
@st.cache_data
def cargar_miembros():
    """Lee miembros_nct.csv y agrega columnas derivadas: subunidad principal,
    país de nacimiento, coordenadas y año de nacimiento."""
    df = pd.read_csv("miembros_nct.csv", encoding="utf-8")

    # La columna FECHA_NACIMIENTO viene en formato DD-MM-AAAA, la convertimos
    # a un objeto datetime real para poder extraer el año fácilmente
    df["FECHA_NACIMIENTO_DT"] = pd.to_datetime(df["FECHA_NACIMIENTO"], format="%d-%m-%Y")
    df["ANIO_NACIMIENTO"] = df["FECHA_NACIMIENTO_DT"].dt.year

    # La columna SUBUNIDADES puede tener varias subunidades separadas por
    # coma (ej. "NCT 127, NCT U"). Definimos la subunidad principal como la
    # primera que no sea "NCT U", ya que NCT U es una unidad rotativa y no
    # una subunidad fija de origen
    def subunidad_principal(texto):
        partes = [p.strip() for p in str(texto).split(",")]
        distintas_de_u = [p for p in partes if p != "NCT U"]
        return distintas_de_u[0] if distintas_de_u else partes[0]

    df["SUBUNIDAD_PRINCIPAL"] = df["SUBUNIDADES"].apply(subunidad_principal)

    # Agregamos país y coordenadas usando el diccionario CIUDADES_INFO
    df["PAIS"] = df["LUGAR DE NACIMIENTO"].apply(lambda c: obtener_info_ciudad(c)["pais"])
    df["LAT"] = df["LUGAR DE NACIMIENTO"].apply(lambda c: obtener_info_ciudad(c)["lat"])
    df["LON"] = df["LUGAR DE NACIMIENTO"].apply(lambda c: obtener_info_ciudad(c)["lon"])

    return df


@st.cache_data
def cargar_videos():
    """Lee videos_nct.csv y convierte la duración de formato MM:SS a
    segundos usando la librería datetime."""
    df = pd.read_csv("videos_nct.csv", encoding="utf-8")

    def duracion_a_segundos(texto):
        # Separamos minutos y segundos a partir del formato "MM:SS"
        minutos, segundos = str(texto).strip().split(":")
        # Usamos timedelta para construir la duración real y extraer los
        # segundos totales de forma precisa
        delta = timedelta(minutes=int(minutos), seconds=int(segundos))
        return delta.total_seconds()

    df["DURACION_SEGUNDOS"] = df["DURACION"].apply(duracion_a_segundos)
    # Creamos también una lista de integrantes por video, separando el texto
    # de la columna MIEMBROS por comas
    df["LISTA_MIEMBROS"] = df["MIEMBROS"].apply(lambda x: [m.strip() for m in str(x).split(",")])

    return df


@st.cache_data
def cargar_letras():
    """Lee letras_nct.csv, cada fila representa una línea cantada por un
    integrante dentro de una canción."""
    df = pd.read_csv("letras_nct.csv", encoding="utf-8")
    return df


# Cargamos los tres datasets una sola vez al iniciar la aplicación
miembros_df = cargar_miembros()
videos_df = cargar_videos()
letras_df = cargar_letras()

# =============================================================================
# CÁLCULOS DERIVADOS QUE SE REUTILIZAN EN VARIAS PÁGINAS
# =============================================================================
@st.cache_data
def calcular_exposicion_por_miembro(videos_df):
    """Calcula el tiempo total de aparición (en segundos) de cada integrante,
    sumando la duración completa de cada video en el que aparece su nombre
    dentro de la columna MIEMBROS. Esto sigue la misma lógica descrita en el
    informe: se asigna la duración total del video a cada integrante listado."""
    exposicion = {}
    for _, fila in videos_df.iterrows():
        for miembro in fila["LISTA_MIEMBROS"]:
            exposicion[miembro] = exposicion.get(miembro, 0) + fila["DURACION_SEGUNDOS"]
    return exposicion


@st.cache_data
def calcular_videos_por_miembro(videos_df):
    """Cuenta en cuántos videos aparece cada integrante."""
    conteo = {}
    for _, fila in videos_df.iterrows():
        for miembro in fila["LISTA_MIEMBROS"]:
            conteo[miembro] = conteo.get(miembro, 0) + 1
    return conteo


@st.cache_data
def calcular_canciones_por_miembro(letras_df):
    """Cuenta en cuántas canciones distintas tiene líneas cada integrante."""
    return letras_df.groupby("MIEMBRO")["CANCION"].nunique().to_dict()


@st.cache_data
def calcular_lineas_por_miembro(letras_df):
    """Cuenta cuántas líneas totales canta cada integrante."""
    return letras_df.groupby("MIEMBRO").size().to_dict()


@st.cache_data
def calcular_participacion_por_subunidad(videos_df, miembros_df):
    """Suma, por subunidad principal de cada integrante, el total de
    apariciones que acumulan sus miembros en todos los videos."""
    mapa_subunidad = dict(zip(miembros_df["MIEMBRO"], miembros_df["SUBUNIDAD_PRINCIPAL"]))
    conteo = {}
    for _, fila in videos_df.iterrows():
        for miembro in fila["LISTA_MIEMBROS"]:
            sub = mapa_subunidad.get(miembro, "Otra")
            conteo[sub] = conteo.get(sub, 0) + 1
    return conteo


exposicion_dict = calcular_exposicion_por_miembro(videos_df)
videos_por_miembro_dict = calcular_videos_por_miembro(videos_df)
canciones_por_miembro_dict = calcular_canciones_por_miembro(letras_df)
lineas_por_miembro_dict = calcular_lineas_por_miembro(letras_df)
participacion_subunidad_dict = calcular_participacion_por_subunidad(videos_df, miembros_df)

# =============================================================================
# MENÚ HORIZONTAL DE NAVEGACIÓN
# =============================================================================
selected = option_menu(
    menu_title=None,
    options=["Inicio", "Conoce NCT", "Estadísticas", "Mapa", "Juego"],
    icons=["house-fill", "people-fill", "bar-chart-line-fill", "geo-alt-fill", "controller"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# =============================================================================
# PÁGINA 1: INICIO
# =============================================================================
if selected == "Inicio":

    st.title("NCT: ¿Expansión Infinita o Desigualdad Masiva?")
    st.caption("Análisis de exposición de contenido y gamificación del sistema de integrantes de NCT")

    col_img, col_texto = st.columns([1, 2])
    with col_img:
        st.image("NCTALLMEMBERS.webp", caption="Miembros de NCT", use_container_width=True)
    with col_texto:
        texto_intro = (
            "El fenómeno global del K-Pop ha transformado la industria musical mediante estructuras de negocio altamente innovadoras. Dentro de este panorama, el megagrupo Neo Culture Technology (NCT), gestionado por la empresa surcoreana SM Entertainment, destaca por implementar un modelo operativo disruptivo denominado “sistema de expansión constante”. Bajo este concepto, el grupo no posee una alineación fija, sino que opera como un ecosistema masivo integrado por más de 20 miembros activos distribuidos en diversas subunidades especializadas como NCT 127, NCT DREAM, WayV y la reciente NCT Wish. Asimismo, coexiste la unidad rotativa NCT U, donde los integrantes colaboran en combinaciones variables por proyecto antes de regresar a sus agrupaciones base.
 "
        )
        st.write(texto_intro)

    st.divider()

    col_obj, col_expl = st.columns(2)
    with col_obj:
        st.subheader("Objetivo")
        st.write(
            "Medir de forma computacional la exposición de cada integrante de NCT "
            "(tiempo en video, líneas cantadas y participación en canciones) para "
            "identificar posibles desigualdades dentro del grupo."
        )
    with col_expl:
        st.subheader("Sobre el proyecto")
        st.write(
            "Esta aplicación procesa datos reales de integrantes, videos y letras para "
            "generar estadísticas interactivas, un mapa de origen internacional del grupo "
            "y un juego para aprender a reconocer a cada integrante."
        )

    st.divider()

    # Tres métricas principales solicitadas para la página de Inicio
    m1, m2, m3 = st.columns(3)
    m1.metric("Integrantes", len(miembros_df))
    m2.metric("Videos analizados", len(videos_df))
    m3.metric("Canciones analizadas", letras_df["CANCION"].nunique())

# =============================================================================
# PÁGINA 2: CONOCE NCT
# =============================================================================
elif selected == "Conoce NCT":

    st.title("Conoce a los integrantes")

    nombre_seleccionado = st.selectbox("Elige un integrante", sorted(miembros_df["MIEMBRO"].unique()))
    fila = miembros_df[miembros_df["MIEMBRO"] == nombre_seleccionado].iloc[0]

    col_foto, col_datos = st.columns([1, 2])
    with col_foto:
        # aquí agregar imagen de Mark (o del integrante seleccionado en cada caso)
        st.info(f"Espacio reservado para la foto de {nombre_seleccionado}")
    with col_datos:
        st.subheader(nombre_seleccionado)
        st.write(f"**Fecha de nacimiento:** {fila['FECHA_NACIMIENTO']}")
        st.write(f"**Edad:** {fila['EDAD']} años")
        st.write(f"**Ciudad de nacimiento:** {fila['LUGAR DE NACIMIENTO']}")
        st.write(f"**Subunidad:** {fila['SUBUNIDADES']}")
        st.write(f"**Posición:** {fila['POSICION']}")
        st.write(f"**Signo zodiacal:** {fila['SIGNO_ZODIACAL']}")

# =============================================================================
# PÁGINA 3: ESTADÍSTICAS
# =============================================================================
elif selected == "Estadísticas":

    st.title("Estadísticas de exposición")

    # -------------------------------------------------------------------
    # FILTROS GENERALES (Subunidad, País, Año, Integrante)
    # -------------------------------------------------------------------
    with st.expander("Filtros", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        subunidades_disp = sorted(miembros_df["SUBUNIDAD_PRINCIPAL"].unique())
        paises_disp = sorted(miembros_df["PAIS"].unique())
        anios_disp = sorted(miembros_df["ANIO_NACIMIENTO"].unique())
        integrantes_disp = sorted(miembros_df["MIEMBRO"].unique())

        filtro_subunidad = f1.multiselect("Subunidad", subunidades_disp, default=subunidades_disp)
        filtro_pais = f2.multiselect("País", paises_disp, default=paises_disp)
        filtro_anio = f3.multiselect("Año de nacimiento", anios_disp, default=anios_disp)
        filtro_integrante = f4.multiselect("Integrante", integrantes_disp, default=integrantes_disp)

    # Aplicamos los filtros sobre la tabla de miembros
    miembros_filtrados = miembros_df[
        miembros_df["SUBUNIDAD_PRINCIPAL"].isin(filtro_subunidad) &
        miembros_df["PAIS"].isin(filtro_pais) &
        miembros_df["ANIO_NACIMIENTO"].isin(filtro_anio) &
        miembros_df["MIEMBRO"].isin(filtro_integrante)
    ]
    nombres_filtrados = set(miembros_filtrados["MIEMBRO"])

    if miembros_filtrados.empty:
        st.warning("No hay integrantes que cumplan con los filtros seleccionados.")
    else:
        tab1, tab2, tab3 = st.tabs(["Tiempo de exposición", "Canciones y líneas", "Videos y subunidades"])

        # -----------------------------------------------------------
        # TAB 1: TIEMPO DE APARICIÓN Y TOP 10 MAYOR / MENOR EXPOSICIÓN
        # -----------------------------------------------------------
        with tab1:
            # Construimos un dataframe con el tiempo de exposición (en minutos)
            # solo de los integrantes que pasan el filtro
            datos_exposicion = {
                "Miembro": [],
                "Minutos": []
            }
            for nombre, segundos in exposicion_dict.items():
                if nombre in nombres_filtrados:
                    datos_exposicion["Miembro"].append(nombre)
                    datos_exposicion["Minutos"].append(round(segundos / 60, 2))

            df_exposicion = pd.DataFrame(datos_exposicion).sort_values("Minutos", ascending=False)

            st.subheader("Tiempo total de aparición")
            fig_tiempo = px.bar(
                df_exposicion, x="Miembro", y="Minutos", color="Miembro",
                title="Minutos totales de aparición en videos por integrante"
            )
            st.plotly_chart(fig_tiempo, use_container_width=True)

            col_top, col_bottom = st.columns(2)
            with col_top:
                st.subheader("Top 10 mayor exposición")
                top10_mayor = df_exposicion.head(10)
                fig_top = px.bar(
                    top10_mayor, x="Miembro", y="Minutos", color="Minutos",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_top, use_container_width=True)
            with col_bottom:
                st.subheader("Top 10 menor exposición")
                top10_menor = df_exposicion.sort_values("Minutos", ascending=True).head(10)
                fig_bottom = px.bar(
                    top10_menor, x="Miembro", y="Minutos", color="Minutos",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_bottom, use_container_width=True)

        # -----------------------------------------------------------
        # TAB 2: CANCIONES Y DISTRIBUCIÓN DE LÍNEAS
        # -----------------------------------------------------------
        with tab2:
            datos_canciones = {
                "Miembro": [],
                "Canciones": []
            }
            for nombre, cantidad in canciones_por_miembro_dict.items():
                if nombre in nombres_filtrados:
                    datos_canciones["Miembro"].append(nombre)
                    datos_canciones["Canciones"].append(cantidad)
            df_canciones = pd.DataFrame(datos_canciones).sort_values("Canciones", ascending=False)

            datos_lineas = {
                "Miembro": [],
                "Lineas": []
            }
            for nombre, cantidad in lineas_por_miembro_dict.items():
                if nombre in nombres_filtrados:
                    datos_lineas["Miembro"].append(nombre)
                    datos_lineas["Lineas"].append(cantidad)
            df_lineas = pd.DataFrame(datos_lineas).sort_values("Lineas", ascending=False)

            col_c, col_l = st.columns(2)
            with col_c:
                st.subheader("Cantidad de canciones")
                if not df_canciones.empty:
                    fig_canciones = px.bar(
                        df_canciones, x="Miembro", y="Canciones", color="Miembro",
                        title="Canciones distintas en las que canta cada integrante"
                    )
                    st.plotly_chart(fig_canciones, use_container_width=True)
                else:
                    st.info("Ningún integrante filtrado tiene líneas registradas.")
            with col_l:
                st.subheader("Distribución de líneas")
                if not df_lineas.empty:
                    fig_lineas = px.pie(
                        df_lineas, names="Miembro", values="Lineas",
                        title="Reparto de líneas cantadas por integrante"
                    )
                    st.plotly_chart(fig_lineas, use_container_width=True)
                else:
                    st.info("Ningún integrante filtrado tiene líneas registradas.")

        # -----------------------------------------------------------
        # TAB 3: VIDEOS POR INTEGRANTE Y PARTICIPACIÓN POR SUBUNIDAD
        # -----------------------------------------------------------
        with tab3:
            datos_videos = {
                "Miembro": [],
                "Videos": []
            }
            for nombre, cantidad in videos_por_miembro_dict.items():
                if nombre in nombres_filtrados:
                    datos_videos["Miembro"].append(nombre)
                    datos_videos["Videos"].append(cantidad)
            df_videos = pd.DataFrame(datos_videos).sort_values("Videos", ascending=False)

            st.subheader("Cantidad de videos")
            if not df_videos.empty:
                fig_videos = px.bar(
                    df_videos, x="Miembro", y="Videos", color="Miembro",
                    title="Cantidad de videos musicales en los que aparece cada integrante"
                )
                st.plotly_chart(fig_videos, use_container_width=True)
            else:
                st.info("Ningún integrante filtrado aparece en los videos analizados.")

            st.subheader("Participación por subunidad")
            # Este gráfico no depende del filtro de integrante individual, pero sí
            # respeta el filtro de subunidad para poder comparar solo las elegidas
            datos_subunidad = {
                "Subunidad": [],
                "Apariciones": []
            }
            for sub, cantidad in participacion_subunidad_dict.items():
                if sub in filtro_subunidad or sub == "Otra":
                    datos_subunidad["Subunidad"].append(sub)
                    datos_subunidad["Apariciones"].append(cantidad)
            df_subunidad = pd.DataFrame(datos_subunidad)
            fig_subunidad = px.pie(
                df_subunidad, names="Subunidad", values="Apariciones",
                title="Total de apariciones en video acumuladas por subunidad"
            )
            st.plotly_chart(fig_subunidad, use_container_width=True)

# =============================================================================
# PÁGINA 4: MAPA
# =============================================================================
elif selected == "Mapa":

    st.title("Origen geográfico de NCT")
    st.caption("Cada marcador representa la ciudad de nacimiento de uno o más integrantes")

    # Filtros propios del mapa
    f1, f2 = st.columns(2)
    subunidades_disp = sorted(miembros_df["SUBUNIDAD_PRINCIPAL"].unique())
    paises_disp = sorted(miembros_df["PAIS"].unique())
    filtro_sub_mapa = f1.multiselect("Subunidad", subunidades_disp, default=subunidades_disp)
    filtro_pais_mapa = f2.multiselect("País", paises_disp, default=paises_disp)

    miembros_mapa = miembros_df[
        miembros_df["SUBUNIDAD_PRINCIPAL"].isin(filtro_sub_mapa) &
        miembros_df["PAIS"].isin(filtro_pais_mapa)
    ]

    # El mapa inicia mostrando el mundo completo, con zoom bajo
    mapa = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    # Usamos MarkerCluster para agrupar visualmente a los integrantes que
    # nacieron en la misma ciudad cuando el mapa está alejado
    cluster = MarkerCluster().add_to(mapa)

    # Agrupamos por ciudad para que, si hay varios integrantes en la misma
    # ciudad, el popup los liste a todos juntos
    for ciudad, grupo in miembros_mapa.groupby("LUGAR DE NACIMIENTO"):
        lat = grupo.iloc[0]["LAT"]
        lon = grupo.iloc[0]["LON"]
        pais = grupo.iloc[0]["PAIS"]

        # Construimos el contenido del popup con nombre y subunidad de cada
        # integrante nacido en esa ciudad
        lineas_popup = "".join(
            f"<b>{fila['MIEMBRO']}</b> - {fila['SUBUNIDAD_PRINCIPAL']}<br>"
            for _, fila in grupo.iterrows()
        )
        popup_html = (
            f"<b>Ciudad:</b> {ciudad}<br><b>País:</b> {pais}<br><hr>{lineas_popup}"
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{ciudad} ({len(grupo)} integrante(s))",
            icon=folium.Icon(color="red", icon="music", prefix="fa")
        ).add_to(cluster)

    st_folium(mapa, width=1200, height=600)

# =============================================================================
# PÁGINA 5: JUEGO - ADIVINA EL INTEGRANTE
# =============================================================================
elif selected == "Juego":

    st.title("Adivina el integrante")
    st.caption("Escribe el nombre artístico del integrante secreto. Tienes 3 intentos por ronda.")

    lista_nombres = list(miembros_df["MIEMBRO"].unique())

    # Orden fijo en el que se van revelando las pistas tras cada error
    ORDEN_PISTAS = ["SUBUNIDAD", "PAIS", "SIGNO"]

    def elegir_nuevo_secreto():
        st.session_state.secreto = random.choice(lista_nombres)
        st.session_state.intentos = 0
        st.session_state.pistas_reveladas = []
        st.session_state.mensaje = ""
        st.session_state.terminado = False

    # Inicializamos el estado del juego la primera vez que se visita la página
    if "secreto" not in st.session_state:
        st.session_state.puntaje = 0
        elegir_nuevo_secreto()

    fila_secreta = miembros_df[miembros_df["MIEMBRO"] == st.session_state.secreto].iloc[0]

    col_score, col_intentos = st.columns(2)
    col_score.metric("Puntaje", st.session_state.puntaje)
    col_intentos.metric("Intentos usados", f"{st.session_state.intentos} / 3")

    # Mostramos las pistas que ya se han revelado
    if st.session_state.pistas_reveladas:
        st.subheader("Pistas")
        for pista in st.session_state.pistas_reveladas:
            st.write(f"🔎 {pista}")

    respuesta = st.text_input("Nombre del integrante", key="respuesta_input", disabled=st.session_state.terminado)

    col_check, col_next = st.columns(2)
    intentar = col_check.button("Comprobar respuesta", disabled=st.session_state.terminado)
    siguiente = col_next.button("Siguiente integrante")

    if intentar:
        # Normalizamos la respuesta del usuario para evitar fallos por
        # mayúsculas o espacios extra
        entrada = respuesta.lower().strip()
        correcto = st.session_state.secreto.lower()

        if entrada == correcto:
            st.success(f"¡Correcto! Era {st.session_state.secreto}. +1 punto")
            st.session_state.puntaje += 1
            elegir_nuevo_secreto()
        elif entrada not in [n.lower() for n in lista_nombres] and entrada != "":
            st.error("Ese nombre no pertenece a ningún integrante de NCT registrado. Intenta de nuevo.")
        else:
            st.session_state.intentos += 1
            if st.session_state.intentos >= 3:
                st.error(f"Se acabaron los intentos. El integrante secreto era {st.session_state.secreto}.")
                st.session_state.terminado = True
            else:
                # Elegimos la siguiente pista disponible según el número de
                # intentos fallidos que lleva el usuario
                tipo_pista = ORDEN_PISTAS[st.session_state.intentos - 1]
                if tipo_pista == "SUBUNIDAD":
                    pista_texto = f"Pertenece a la subunidad: {fila_secreta['SUBUNIDAD_PRINCIPAL']}"
                elif tipo_pista == "PAIS":
                    pista_texto = f"Nació en: {fila_secreta['PAIS']}"
                else:
                    pista_texto = f"Su signo zodiacal es: {fila_secreta['SIGNO_ZODIACAL']}"
                st.session_state.pistas_reveladas.append(pista_texto)
                st.warning("Respuesta incorrecta. Nueva pista revelada.")

    if siguiente:
        elegir_nuevo_secreto()

    if st.session_state.terminado:
        st.info("Presiona 'Siguiente integrante' para comenzar una nueva ronda.")
