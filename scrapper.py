from selenium import webdriver
from bs4 import BeautifulSoup
import chromedriver_binary
from googletrans import Translator, constants
import json 
import csv 

driver = webdriver.Chrome()
driver.get('http://facundoq.github.io/datasets/lsa64/')
translator =  Translator()

content = driver.page_source
soup = BeautifulSoup(content)

signs = []

sign = {}
for i,td in enumerate(soup.find_all('td')):
    tableHead = i % 3
    if td.text != '':
        if tableHead == 0:        
            sign['id'] = td.text
        elif tableHead == 1:
            if td.text == 'Skimmer':
                sign['name'] = 'Skimmer'
            elif td.text == 'Son':
                sign['name'] = 'Hijo'
            elif td.text == 'Mock':
                sign['name'] = 'Burlarse de'
            else:
                sign['name'] = translator.translate(td.text, dest='es').text
        elif tableHead == 2:
            sign['handsUsed'] = td.text
            signs.append(sign)
            sign = {}
signs.sort(key=lambda s:s['id'])

with open('dataset.csv','w', encoding='UTF8', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(signs[0].keys())
    for s in signs:
        writer.writerow(s.values())




#with open('dataset.json', 'w') as outfile:
#    json.dump(signs,outfile)