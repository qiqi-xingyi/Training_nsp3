#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=ht3_aim -A ht3_aim
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N ESM1b
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e error.err
#PBS -o log.log
### Only send mail when job is aborted or terminates abnormally
### Number of nodes
#PBS -l nodes=1:ppn=4:gpus=1
### Memory
#PBS -l mem=16gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=06:00:00
  
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Load all required modules for the job
module load tools
module load anaconda3/2020.07

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/services/tools/anaconda3/2020.07/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/services/tools/anaconda3/2020.07/etc/profile.d/conda.sh" ]; then
        . "/services/tools/anaconda3/2020.07/etc/profile.d/conda.sh"
    else
       	export PATH="/services/tools/anaconda3/2020.07/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export LD_LIBRARY_PATH=/home/projects/ht3_aim/people/erikie/NSPThesis/venv/lib:$LD_LIBRARY_PATH

conda activate /home/projects/ht3_aim/people/erikie/NSPThesis/venv

cd ../nsp3

python setup.py install

#nsp3_client train -c experiments/nsp3_client/ESM1b/ESM1b.yml
nsp3 train -c experiments/nsp3/ESM1b/ESM1b_finetune.yml
