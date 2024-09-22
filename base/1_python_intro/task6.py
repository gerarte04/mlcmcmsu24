def check(s, filename):
    arr = {}

    for sub in s.split():
        subs = sub.lower()
        if subs not in arr:
            arr[subs] = 0
        arr[subs] += 1
    
    with open(filename, "w") as fd:
        for key in sorted(arr.keys()):
            fd.write("{} {}\n".format(key, arr[key]))
        fd.close()
