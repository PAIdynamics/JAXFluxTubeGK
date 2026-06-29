def create_mms_assertion_script(geometries, resolutions, runtime_modes, \
                                is_mms_by_parts=False, outfile_suffix=None):
    modes = {}
    modes["cpu"] = ["cpu"]
    modes["gpu"] = [mode for mode in runtime_modes if mode != "cpu"]

    if modes["gpu"]:
        ptypes = ["cpu", "gpu"]
    else:
        ptypes = ["cpu"]

    broad_template = 'python ../mms_assert.py --id <ID> ' + \
                     '<PATH>/reference.out ' + \
                     'mms_broad_<PTYPE>.out 10e-15 10e-5'

    deep_template = 'python ../../mms_assert.py --id <ID> ' + \
                     '<PATH>/reference.out ' + \
                     'mms_deep_${MODE}.out 10e-15 10e-5'

    script = {}
    script["broad"] = {}
    script["deep"]  = {}

    for ptype in ptypes:
        for geom in geometries:
            script["deep"][geom] = {}

    for ptype in ptypes:
        if is_mms_by_parts:
            bline = broad_template.replace("<PTYPE>",
                                           ptype + '_' + outfile_suffix)
        else:
            bline = broad_template.replace("<PTYPE>", ptype)
        script["broad"][ptype] = ['#!/bin/bash -l']
        script["broad"][ptype].append('')

        bcounter = 1
        for geom in geometries:
            ref_path = geom + "/resolution_1"
            for mode in modes[ptype]:
                sline = bline.replace("<ID>", str(bcounter))
                sline = sline.replace("<PATH>", ref_path)
                script["broad"][ptype].append( \
                    'echo -e "\033[1;34mAsserting the broad MMS ' + mode + \
                    ' output of ' + geom + '\033[0m"')
                script["broad"][ptype].append(sline)
                bcounter += 1
            script["broad"][ptype].append('echo ""')
            script["broad"][ptype].append('')

    if is_mms_by_parts:
        return script["broad"]["gpu"]

    dline = deep_template
    for geom in geometries:
        script["deep"][geom] = ['#!/bin/bash -l']
        script["deep"][geom].append('')
        script["deep"][geom].append('if [ $# -eq 0 ]; then')
        script["deep"][geom].append('    MODE=cpu')
        script["deep"][geom].append('elif [ $# -eq 1 ]; then')
        script["deep"][geom].append('    MODE=$1')
        script["deep"][geom].append('else')
        script["deep"][geom].append(\
        '    echo -e "\033[0;31mToo many arguments. Pass string specifying ' + \
        'the runtime mode!\033[0m"')
        script["deep"][geom].append(\
        '    echo -e "\033[0;31mi.e. cpu, cxx, acc, ompx\033[0m"')
        script["deep"][geom].append('    return 1')
        script["deep"][geom].append('fi')
        script["deep"][geom].append('')

        dcounter = 1
        for res in resolutions:
            ref_path = "resolution_" + str(res)
            sline = dline.replace("<ID>", str(dcounter))
            sline = sline.replace("<PATH>", ref_path)
            script["deep"][geom].append( \
                'echo "Asserting the deep MMS ' + geom + ' ${MODE} ' + \
                'output of ' + 'resolution ' + str(res) + '"')
            script["deep"][geom].append(sline)
            dcounter += 1
            script["deep"][geom].append('')


    return script
