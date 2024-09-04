from bs4 import BeautifulSoup
import requests
locations = ['17821', '17925', '18037', '17975', '17898', '17972']


data = []
for location in locations:
    print(f"Scraping location {location}")
    r = requests.get(f"https://www.hemnet.se/salda/bostader?location_ids[]={location}&sold_age=10m", headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(r.content, 'html5lib')
    soup = BeautifulSoup(r.content, 'html5lib')
    cards = soup.find_all(class_='hcl-card')
    for card in cards:
        name = card.find(class_='Header_truncate__ebq7a')
        Saledate = card.find(class_='Label_hclLabelSoldAt__gw0aX')
        SizeData = card.find_all(class_='Text_hclText__V01MM')
        SellingData = card.find(class_="SellingPriceAttributes_contentWrapper__VaxX9")
        LocationData = card.find(class_='Location_address___eOo4')
        TypeOfProperty = card.find(class_='hcl-icon')
        Size = ''
        Room = ''
        Fee = ''
        Endprice = ''
        City = ''
        District = ''
        #print(card.prettify())
        if name:
            name = name.text
            if Saledate:
                Saledate = Saledate.text.split("åld")[-1]
            if SizeData:
                Size = SizeData[0].text.strip(' m²').replace(",",".")
                Room = SizeData[1].text.strip(' rum').replace(",",".")
                FeeOrLawn = SizeData[2].text
                if 'kr' in FeeOrLawn:
                    Fee = FeeOrLawn.strip(' kr/mån')
            if SellingData:
                Endprice = SellingData.find(class_='hcl-flex--container hcl-flex--gap-2 hcl-flex--justify-space-between hcl-flex--md-justify-flex-start')
                if Endprice:
                    Endprice = Endprice.text
                    Endprice = Endprice.replace(" ", "").replace(" ", "").strip("Slutpris").split("kr")[0].replace("k", "")
            if LocationData:
                LocationData = LocationData.text
                splitData = LocationData.split(',')
                City = splitData[-1]
                District = splitData[0]
            if TypeOfProperty:
                TypeOfProperty = TypeOfProperty.text
            data.append([name, Size, Room, Fee, Endprice, 0, Saledate, City, District, TypeOfProperty])

with open("housedata2022.csv", "w") as f:
    f.write("name, Size, Room, Fee, Endprice, KvMPrice, Saledate, City, District, TypeOfProperty\n")
    for row in data:
        stringdata = f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{row[7]},{row[8]},{row[9]}\n".replace(" ", "")
        f.write(stringdata)
