#!/bin/sh
git status  
git add *  
git commit -m 'add some code from lenevo'
# git commit -m 'add some results from Server'
git pull --rebase origin master   #download data
git push origin master            #upload data
git stash pop