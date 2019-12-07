import urllib.request
import sys
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import requests
import random
import os
from bs4 import BeautifulSoup
import re


URL="https://www.flickr.com/explore"
# r=requests.get(URL)

# with open("a.html", "w") as file:
# 	file.write(r.text)
# # sys.exit(0)
# soup=BeautifulSoup(r.content,'html.parser')

# # print(soup)
# view_as=soup.find('div', attrs = {'class':'view photo-list-view requiredToShowOnServer'})
# for image_view in soup.find_all('div', attrs = {'class':'view photo-list-photo-view requiredToShowOnServer awake'}):
# 	# print(image_view)
# 	interaction_view=image_view.find('a')
# 	# print(interaction_view)
# 	# image_html=interaction_view.find('div',attrs={'class':'photo-list-photo-interaction'})
# 	# print(image_html.a['href'])

# ass = soup.find_all(href=re.compile(r'/photos/iainleach/49039766327/in/explore-2019-11-09/'))
# print(ass)

driver = webdriver.Firefox()
driver.get(URL)
OUT_DIR="./flickr/"
if not os.path.exists(OUT_DIR):
	os.makedirs(OUT_DIR)

idx=1
image_links=set()
SCROLL_PAUSE_TIME = 2
NO_OF_IMAGES=2
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while len(image_links)<NO_OF_IMAGES:


	for image_link in driver.find_elements_by_class_name('overlay'):
		image_links.add(image_link.get_attribute('href'))
	# Scroll down to bottom
	driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

	# Wait to load page
	time.sleep(SCROLL_PAUSE_TIME)

	# Calculate new scroll height and compare with last scroll height
	new_height = driver.execute_script("return document.body.scrollHeight")
	if new_height == last_height:
		break
	last_height = new_height
	


image_download_links=[]
for image_link in image_links:
	driver.get(image_link)
	views=driver.find_element_by_class_name('view-count-label')
	fave=driver.find_element_by_class_name('fave-count-label')
	faves_score=float(fave.text.replace(',',''))/float(views.text.replace(',',''))
	# image_download=driver.find_element_by_class_name('view photo-engagement-view')
	time.sleep(2)
	image_download_link=driver.find_element_by_class_name('download').find_element_by_tag_name('a').get_attribute('href')

	image_download_link=re.sub('/./','/h/',image_download_link)
	image_download_links.append(image_download_link)


	driver.get(image_download_link)
	temp.sleep(1)
	temp=driver.find_element_by_id('allsizes-photo')
	link=temp.find_element_by_tag_name('img')
	download_link=link.get_attribute('src')
	urllib.request.urlretrieve(download_link, os.path.join(OUT_DIR,'flickr_'+str(idx)+'_'+str(faves_score)+'.jpg'))	
	idx+=1


driver.close()
