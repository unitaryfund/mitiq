#!/usr/bin/env python

import os
import json
import re

exclude_dirs = ['tests']
# a list of exclude patterns for decrapated or poorly named
# internal functions
exclude_patterns = [r'.*_td', r'.*_es', r'.*_mc', r'.*_ode', r'_.*']
mitiq_root = "../../mitiq/"
colors = ["#0B5FA5", "#043C6B", "#3F8FD2", # blue colors
          "#00AE68", "#007143", "#36D695", # green colors
          "#FF4500", "#692102", "#BF5730"
          ]
## additional colors:
#          "#FF9400", "#A66000", "#FFAE40"
#          "#FF6F00", "#A64800", "#BF6E30"
## specifiy code tree structure:
module_cmap = {"zne":               0, # zne
               "factories":         1, # factories
               "pyquil_utils":      2, # APIs: pyquil
               "mitiq_qiskit":      3, # APIs: qiskit
               "qiskit_utils":      3, # APIs: qiskit
               "conversions":       3,
               "folding":           4,
               "benchmarks":        5, # benchmarks
               "random_circ":       5,
               }
# add here hidden modules
hidden_modules = ['benchmarks', 'mitiq_pyquil', 'mitiq_qiskit']
module_list  = []
num_items = 0

for root, dirs, files in os.walk(mitiq_root):
  if not ".svn" in root and root == "../../mitiq/":
    print("Checking files...")
    # search modules in .py files
    for f in files:
      # check these are only .py files, not private modules and not the setup.py
      if f[-3:] == ".py" and f[0] != "_" and f != "setup.py":
        module = f[:-3]
        if module not in hidden_modules:
          idx   = module_cmap[module] if module in module_cmap else -1
          color = colors[idx] if idx >= 0 else "black"
          symbol_list = []
          cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (mitiq_root, f)
          for name in os.popen(cmd).readlines():
            if not any([re.match(pattern, name) for pattern in exclude_patterns]):
              symbol_list.append({"name": name.strip(), "size": 1000, "color": color})
              num_items+=1
        module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})
    print("... files checked! Found {} items.".format(num_items))
    # search modules in directories:
    print("Now checking directories...")
    for d in dirs:
      #print("dirs ", dirs)
      if d in hidden_modules:
        print("directory ",d)
        for root, dr, files in os.walk(mitiq_root+'/'+d):
          for f in files:
            # check these are only .py files, not private modules, not setup.py
            # check also that file does not start with "test"
            if f[-3:] == ".py" and f[0] != "_" and f != "setup.py" and f[:4] != "test":
              print("file ",f)
              module = f[:-3]
              idx   = module_cmap[module] if module in module_cmap else -1
              color = colors[idx] if idx >= 0 else "black"
              symbol_list = []
              cmd = "egrep '^(def|class) ' %s/%s | cut -f 2 -d ' ' | cut -f 1 -d '('" % (mitiq_root+'/'+d, f)
              #print((mitiq_root+'/'+d, f))
              for name in os.popen(cmd).readlines():
                if not any([re.match(pattern, name) for pattern in exclude_patterns]):
                  #print("pattern ", exclude_patterns)
                  #print("name ", name)
                  symbol_list.append({"name": name.strip(), "size": 1000, "color": color})
                  num_items+=1
              module_list.append({"name": module, "children": symbol_list, "color": color, "idx": idx})
    print("... directories checked! Found {} items.".format(num_items))
module_list_sorted = sorted(module_list, key=lambda x: x["idx"])
mitiq_struct = {"name": "mitiq", "children": module_list_sorted, "size": 2000}
# write to file as a json database
with open('d3_data/mitiq.json', 'w') as outfile:
    json.dump(mitiq_struct, outfile, sort_keys=True, indent=4)
# print the number of items found
print(num_items)