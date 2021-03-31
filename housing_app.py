from bs4 import BeautifulSoup
import requests
import numpy as np


def scrape(num_pages=18):
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

        # liczenie div o klasie row row

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
                one_site_rooms = np.append(one_site_rooms, np.random.randint(3, 6))
                if (len(one_site_rooms) % 35) == 0:
                    break
            print("counter_rooms_li =", counter_rooms_li)
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
    return all_prices_per_m2.reshape(1, -1)[0], all_rooms.reshape(1, -1)[0], all_areas.reshape(1, -1)[0]


if __name__ == "__main__":

    result = scrape()

    print()
    all_prices_per_m2 = result[0]
    print("all_prices_per_m2 =")
    print(all_prices_per_m2)
    print("Ilosc: ", len(all_prices_per_m2))

    print()
    all_rooms = result[1]
    print("all_rooms =")
    print(all_rooms)
    print("Ilosc: ", len(all_rooms))

    print()
    all_areas = result[2]
    print("all_areas =")
    print(all_areas)
    print("Ilosc: ", len(all_areas))

    # lista cen calkowitych policzonych przeze mnie
    print()
    all_prices_moje = np.array([])
    for i in range(len(all_areas)):
        all_prices_moje = np.append(all_prices_moje, all_areas[i]*all_prices_per_m2[i])
    print("all_prices_moje =")
    print(all_prices_moje)
    print("Ilosc:", len(all_prices_moje))

    for i in range(max(len(all_prices_per_m2), len(all_rooms), len(all_areas))):
        print(i+1)
        print(all_prices_per_m2[i], all_rooms[i], all_areas[i], all_prices_moje[i])
    
    print()
    print("Ostatnia:")
    print(all_prices_per_m2[-1], all_rooms[-1], all_areas[-1], all_prices_moje[-1])
