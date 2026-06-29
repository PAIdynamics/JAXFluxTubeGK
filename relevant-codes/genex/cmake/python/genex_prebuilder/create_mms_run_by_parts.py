def create_mms_run_by_parts(partition_names):

    mms_on_gpu = [str(i) for i, e in enumerate(partition_names)
                     if e == 'gpu' or e == 'apu']
    mms_others    = [str(i) for i, e in enumerate(partition_names)
                     if not (e == 'gpu' or e == 'apu')]

    script = ['#!/bin/bash -l']
    script.append('')
    script.append('rm -f *.done')
    script.append('')

    par_commands = ''
    for i in mms_on_gpu:
        command = 'sbatch --wait submit_mms_broad_gpu_' + i + '.sh & '
        script.append('echo -e "\033[1;34m' + command[:-1] + '\033[0m"')
        par_commands += command
    script[-1] = script[-1][:-5] + ' wait\033[0m"'
    par_commands = par_commands + 'wait'

    script.append('command="' + par_commands + '"')
    script.append('eval $command')
    script.append('echo ""')
    script.append('')

    for i in mms_on_gpu:
        script.append('echo -e "\033[1;36mAsserting MMS GPU part ' + \
                      i + '!\033[0m"')
        script.append('source ./mms_assert_broad_gpu_' + i + '.sh')
        script.append('if [ -f "mms_broad_gpu_' + i + '.done" ]; then')
        script.append('    rm mms_broad_gpu_' + i + '.done')
        script.append('else')
        script.append('    echo -e "\033[1;31mError: MMS GPU part ' + i + \
                      ' was prematurely terminated!\033[0m" >&2')
        script.append('fi')
        script.append('echo ""')
        script.append('')

    return script
