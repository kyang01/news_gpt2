import ujson as json
import os
import argparse
import ipdb
from tqdm import tqdm
import pandas as pd

sites = ["pjmedia.com", "dailykos.com", "breitbart.com", "forward.com", "alternet.org", "bipartisanreport.com", "wnd.com", "breitbart.com", "dailywire.com", "theblaze.com", "redstate.com", "infowars.com", "bigleaguepolitics.com", "dailycaller.com"]
f_out = open("misinfonews.jsonl", "w")
with open("realnews.jsonl", 'r') as f:
	domains = {}
	for l_no, line in tqdm(enumerate(f)):
		article = json.loads(line)
		if any(site in article["domain"] for site in sites):
			f_out.write(line)
f_out.close()

articles = []
with open("misinfonews.jsonl", 'r') as f:
	for l_no, line in tqdm(enumerate(f)):
		articles.append(json.loads(line))
articles = pd.DataFrame(articles)
articles.to_csv("misinfo.csv")