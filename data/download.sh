#!/bin/bash
if [[ -d train ]] && [[ -d dev ]]; then
  echo "data exist"
  exit 0
else
  wget -c http://fake_url/to/fake_data.zip
fi

if [[ `md5sum -c md5sum.txt` =~ 'OK' ]] ; then
    unzip fake_data.zip
else
  echo "download data error!" >> /dev/stderr
  exit 1
fi
