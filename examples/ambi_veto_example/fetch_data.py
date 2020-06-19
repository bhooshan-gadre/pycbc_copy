import json
try:
    from urllib.request import urlretrieve
except ImportError:  # python < 3
    from urllib import urlretrieve

url_head = "https://www.gw-openscience.org/eventapi/json/GWTC-1-confident/GW150914/v3/"
lfile = "L-L1_GWOSC_4KHZ_R1-1126257415-4096.gwf"
hfile = "H-H1_GWOSC_4KHZ_R1-1126257415-4096.gwf"

print("Downloading from {} ".format(url_head))
url = url_head + lfile
urlretrieve(url, filename=lfile)
print("Done with {}".format(lfile))
url = url_head + hfile
urlretrieve(url, filename=hfile)
print("Done with {}".format(hfile))


# # $ FrChannels L-L1_GWOSC_4KHZ_R1-1126257415-4096.gwf
# L1:GWOSC-4KHZ_R1_STRAIN 4096
# L1:GWOSC-4KHZ_R1_DQMASK 1
# L1:GWOSC-4KHZ_R1_INJMASK 1
# # $ FrChannels H-H1_GWOSC_4KHZ_R1-1126257415-4096.gwf
# H1:GWOSC-4KHZ_R1_STRAIN 4096
# H1:GWOSC-4KHZ_R1_DQMASK 1
# H1:GWOSC-4KHZ_R1_INJMASK 1
