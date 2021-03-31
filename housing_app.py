from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def scrape_and_save_to_df(num_pages=18):
    base_url = "http://morizon.pl/mieszkania/najtansze/bialystok/?page="

    # sumaryczne listy cen, pokoi i powierzchni
    all_prices_per_m2 = np.array([])
    all_rooms = np.array([])
    all_areas = np.array([])

    for page_num in range(1, num_pages + 1):

        # kazda podstrona ma swoja liste cen, pokoi i powierzchni
        one_site_prices_per_m2 = np.array([])
        one_site_rooms = np.array([])
        one_site_areas = np.array([])

        print("souping page:", page_num)
        url = base_url + str(page_num)
        soup = BeautifulSoup(requests.get(url).text, 'html.parser')

        for p in soup("p", {"class": [
                            "single-result__price single-result__price--currency",
                            "single-result__price single-result__price--ask"]}):
            # rsplit(' ', 1)[0] bierze ostatnia spacje i to co jest przed nia.
            # replace(' ', '') zamienia na przyklad 10 500,28 na 10500,28
            # split(',')[0] z 10500,28 pobiera 10500
            # Jesli nie da sie zrzutowac string "Zapytaj o cene" na int to rzuci wyjatek
            try:
                one_site_prices_per_m2 = np.append(one_site_prices_per_m2,
                                                   float(p.get_text().rsplit(' ', 1)[0].replace(' ', '').replace(',', '.')))
                # print(one_site_prices_per_m2)
            except ValueError as e:
                print("Nie podano ceny dla", len(one_site_prices_per_m2))
                # przypisuje do listy cene za m2 z poprzedniego mieszkania
                one_site_prices_per_m2 = np.append(one_site_prices_per_m2, one_site_prices_per_m2[len(one_site_prices_per_m2)-1])
        # do calej listy dodaje liste z jednej podstrony
        all_prices_per_m2 = np.append(all_prices_per_m2, one_site_prices_per_m2)


        for ul in soup("ul", {"class": "param list-unstyled list-inline"}):
            # licznik w ilu li jest pokoj
            counter_rooms_li = 0
            for li in ul.findAll("li"):
                if "pok" in li.get_text():
                    counter_rooms_li += 1
                    for b in li.findAll("b"):
                        one_site_rooms = np.append(one_site_rooms, int(b.get_text()))
            # gdy nie podano ilosci pokoi
            if counter_rooms_li == 0:
                # jesli nie podano ilosci pokoi to losuje sposrod 3, 4, 5
                one_site_rooms = np.append(one_site_rooms, np.random.randint(3, 6))
                if (len(one_site_rooms) % 35) == 0:
                    break
            # print("counter_rooms_li =", counter_rooms_li)
            if (len(one_site_rooms) % 35) == 0:
                break
        # do calej listy dodaje liste z jednej podstrony
        all_rooms = np.append(all_rooms, one_site_rooms)

        for ul in soup("ul", {"class": "param list-unstyled list-inline"}):
            for li in ul.findAll("li"):
                if "m" in li.get_text():
                    for b in li.findAll("b"):
                        one_site_areas = np.append(one_site_areas, int(b.get_text()))
            if (len(one_site_areas) % 35) == 0:
                break
        # do calej listy dodaje liste z jednej podstrony
        all_areas = np.append(all_areas, one_site_areas)

    # zwraca wyplaszczone listy
    all_prices_per_m2 = all_prices_per_m2.reshape(1, -1)[0]
    all_rooms = all_rooms.reshape(1, -1)[0]
    all_areas = all_areas.reshape(1, -1)[0]

    # lista cen calkowitych policzonych przeze mnie
    print()
    all_prices = np.array([])
    for i in range(len(all_areas)):
        all_prices = np.append(all_prices, all_areas[i] * all_prices_per_m2[i])

    df = pd.DataFrame({"all_rooms": all_rooms,
                       "all_areas": all_areas,
                       "all_prices": all_prices})

    df["all_rooms"] = df["all_rooms"].astype("int")
    df["all_areas"] = df["all_areas"].astype("int")

    return df

def create_model(df):

    # ponizej przygotowanie do grid search, ale na dole wzialem tylko najlepszy
    # X = df[["all_rooms", "all_areas"]]
    # y = df["all_prices"]

    # from sklearn.model_selection import train_test_split

    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # from sklearn.ensemble import RandomForestRegressor

    # domyslnie nie ma max_depth czyli przetrenowany
    # reg = RandomForestRegressor()
    # reg.fit(X_train, y_train)

    # wspolczynnik R kwadrat
    # reg_score = reg.score(X_test, y_test)
    # print("Przed grid search r2 score wynosi", reg_score)

    # from sklearn.model_selection import GridSearchCV
    #
    # param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30],
    #                'min_samples_leaf': [3, 4, 5, 10, 15]}]


    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)
    # metoda oceny to R kwadrat
    # gs = GridSearchCV(model, param_grid=param_grid, scoring='r2')
    # gs.fit(X_train, y_train)

    # gs_score = gs.score(X_test, y_test)
    # print("Po grid search r2 score wynosi", gs_score)

    # model = gs.best_estimator_
    # print("model")
    # print(model)

    # po zastosowaniu grid search wybralem najlepszy model
    X = df[["all_rooms", "all_areas"]]
    y = df["all_prices"]

    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(max_depth=4, min_samples_leaf=5)
    model.fit(X, y)

    return model

# poczatek dash
# style zewnetrzne
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# nazwa __name__ to zmienna środowiskowa
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# potrzebne do heroku
server = app.server

df = scrape_and_save_to_df()
model = create_model(df)

print("df =")
print(df)
print(df.info())

app.layout = html.Div([
    html.Div([
        html.H3('Regresyjny Model Uczenia Maszynowego - RandomForestRegressor.'),
        html.H3('Przewidywanie ceny mieszkania w Białymstoku.'),
        html.H6('Danymi wejściowymi są ilość pokoi oraz powierzchnia w metrach kwadratowych.'),
        html.H6('Dane oraz ceny potrzebne do trenowania modelu są scrapowane z portalu morizon.pl'),
        html.H6('z pierwszych 18 podstron (na każdej 35 ofert) posortowane od najniższej ceny.'),
        html.Div([
            html.A('http://morizon.pl/mieszkania/najtansze/bialystok/'),
            ], style={'background-color': "white"}
        )
    ], style={'textAlign': 'center'}),
    html.Hr(),
    html.Div([
        html.Label('Podaj ilość pokoi:'),
        dcc.Slider(
            id='slider-1',
            min=df["all_rooms"].min(),
            max=df["all_rooms"].max(),
            step=1,
            marks={i: str(i) for i in range(df["all_rooms"].min(), df["all_rooms"].max() + 1)}
        ),
        html.Hr(),
        html.Label('Podaj liczbę metrów kwadratowych:'),
        dcc.Slider(
            id='slider-2',
            min=20,
            max=100,
            step=1,
            marks={i: str(i) for i in range(20, 101, 10)},
            tooltip={'placement': 'bottom'}
        ),

        html.Div([
            html.Hr(),
            html.H4('Podano parametry:'),
            html.Div(id='div-1'),
            html.Div(id='div-2'),
            html.Hr()
        ], style={'margin': '0 auto', 'textAlign': 'center'})

    ], style={'width': '80%', 'textAlign': 'left', 'margin': '0 auto', 'fontSize': 22,
              'background-color': 'white', 'padding': '30px'})
], style={'background-color': '#AF90C2'})


@app.callback(
    Output('div-1', 'children'),
    [Input('slider-1', 'value'),
     Input('slider-2', 'value')]
)
def display_parameters(val1, val2):
    if val1 and val2:
        return html.Div([
            html.H6(f'Ilość pokoi: {val1}'),
            html.H6(f'Rozmiar w metrach kwadratowych: {val2}')
        ], style={'textAlign': 'left'})
    else:
        return html.Div([
            html.H6('Podaj wszystkie parametry.')
        ])

@app.callback(
    Output('div-2', 'children'),
    [Input('slider-1', 'value'),
     Input('slider-2', 'value')]
)
def predict_value(val1, val2):
    if val1 and val2:

        df_sample = pd.DataFrame(
            data=[
                [val1, val2]
            ],
            columns=['all_rooms', 'all_areas']
        )
        print(df_sample)

        price = model.predict(df_sample)[0]
        price = round(price)

        return html.Div([
            html.H4(f'Przewidywana cena: {price} zł')
        ], style={'background-color': '#AF90C2', 'width': '60%', 'margin': '0 auto'})


# if __name__ == '__main__':
#     app.run_server(debug=True)


