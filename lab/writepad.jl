using JSON
using DataFrames
using Gadfly
using DecisionTree

conf_os=open("../conf.json", "r")
conf=JSON.parse(readall(conf_os))
close(conf_os)
sat = readtable(conf["dpath"]*"/sat/sat.data", separator=' ', header=false)
plot(sat, x="x37", Geom.histogram)
features=convert(Array, sat[1:36])
labels=convert(Array, sat[37])
tr=build_tree(labels, features)
tr=prune_tree(tr, 0.9)

sat_tst=readtable(conf["dpath"]*"/sat/sat-test.data", separator=' ', header=false)
cntarray=zeros(6,6)
size_tst=size(sat_tst, 1)
for i in 1:size_tst
    ans=sat[i,37]
    res=apply_tree(tr, convert(Array, sat[i, 1:36]))[1]
    if ans==7
        ans-=1
    end
    if res==7
        res-=1
    end
    cntarray[ans, res]+=1
end
println(cntarray)
