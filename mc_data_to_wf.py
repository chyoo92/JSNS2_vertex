import rat
import numpy as np
import argparse




parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--energy', action='store', type=int, default=0)
parser.add_argument('--num', action='store', type=int, default=0)
args = parser.parse_args()
# "/store/hep/users/yewzzang/JSNS2/rat_mc_positron/positron_"+str(j+1)+"MeV/positron_"+str(j+1)+
# /store/hep/users/yewzzang/JSNS2/rat_mc_positron/positron_10MeV/positron_10MeV_0.root   0 to 100


path = args.input
ds_reader = rat.dsreader(path)


high_wf = []
low_wf = []
high_time = []
low_time = []
for event in ds_reader:
    ev = event.GetEV(0)
    n_pmt = ev.GetPMTCount()
    for ipmt in range(n_pmt):
        pmt = ev.GetPMT(ipmt)

        high_gain_waveform = pmt.GetHighGainWaveform()
        low_gain_waveform = pmt.GetLowGainWaveform()
        high_gain_samples = np.array(high_gain_waveform.GetSamples())
        low_gain_samples = np.array(low_gain_waveform.GetSamples())
        

        hg_sampling_time = high_gain_waveform.GetSamplingTime()
        lg_sampling_time = low_gain_waveform.GetSamplingTime()


        high_wf.append(high_gain_samples)
        low_wf.append(low_gain_samples)
        high_time.append(hg_sampling_time)
        low_time.append(lg_sampling_time)


print(args.energy,args.input)
np.savetxt('mc_waveform/high_wf_E'+str(args.energy)+'_'+str(args.num)+'.csv',high_wf,delimiter=",")
np.savetxt('mc_waveform/low_wf_E'+str(args.energy)+'_'+str(args.num)+'.csv',high_wf,delimiter=",")
np.savetxt('mc_waveform/high_time_E'+str(args.energy)+'_'+str(args.num)+'.csv',high_time,delimiter=",")
np.savetxt('mc_waveform/low_time_E'+str(args.energy)+'_'+str(args.num)+'.csv',high_time,delimiter=",")