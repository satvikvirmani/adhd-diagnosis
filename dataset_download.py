import requests

urls = [
    "https://ieee-dataport.s3.amazonaws.com/ieee-dataport/open/28547/ADHD_part1.zip",
    "https://ieee-dataport.s3.amazonaws.com/ieee-dataport/open/28547/ADHD_part2.zip",
    "https://ieee-dataport.s3.amazonaws.com/ieee-dataport/open/28547/Control_part1.zip",
    "https://ieee-dataport.s3.amazonaws.com/ieee-dataport/open/28547/Control_part2.zip",
    "https://ieee-dataport.s3.amazonaws.com/ieee-dataport/open/28547/Standard-10-20-Cap19new.zip",
    "https://ieee-dataport.s3.amazonaws.com/ieee-dataport/open/28547/Channel_Labels.zip"
]

for url in urls:
    filename = url.split("?")[0].split("/")[-1]
    response = requests.get(url, stream=True)
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Downloaded:", filename)