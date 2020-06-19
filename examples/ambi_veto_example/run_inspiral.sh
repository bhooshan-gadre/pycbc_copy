
seg_duration=256
jump=$((seg_duration / 2))
event_time=1126259462

gps_start=$((event_time - jump))
gps_end=$((event_time + jump))
bank="/work/bhooshan.gadre/ambi_chisq/test/banks/pico_bank.hdf"
veto_bank="/work/bhooshan.gadre/ambi_chisq/test/banks/mini_bank.hdf"

ifos='H1 L1'
for ifo in $ifos
do
    site=${ifo:0:1}
    echo 'site is' $site
    `which pycbc_inspiral` --pad-data 16 --strain-high-pass 20 --sample-rate 512 \
        --segment-length 256 --segment-start-pad 112 --segment-end-pad 16 \
        --allow-zero-padding  --taper-data 1 --psd-estimation median \
        --psd-segment-length 16 --psd-segment-stride 8 --psd-inverse-length 16 \
        --psd-num-segments 30 --autogating-threshold 50 \
        --autogating-cluster 0.1 --autogating-width 0.25 --autogating-taper 0.25 \
        --autogating-pad 16 --autogating-max-iterations 5 --low-frequency-cutoff 25 \
        --approximant 'SPAtmplt:mtotal<4' 'SEOBNRv4_ROM:else' --order -1 \
        --snr-threshold 3.0 --keep-loudest-interval 1.072 --keep-loudest-num 100 \
        --keep-loudest-stat newsnr_sgveto --cluster-method window --cluster-window 1 \
        --cluster-function symmetric --chisq-snr-threshold 3. --chisq-bins \
        "0.72*get_freq('fSEOBNRv4Peak',params.mass1,params.mass2,params.spin1z,params.spin2z)**0.7" \
        --newsnr-threshold 3.0 --sgchisq-snr-threshold 5.0 --sgchisq-locations \
        "mtotal>30:20-15,20-30,20-45,20-60,20-75,20-90,20-105,20-120" \
        --finalize-events-template-rate 500 \
        --processing-scheme cpu:4 --channel-name ${ifo}:GWOSC-4KHZ_R1_STRAIN \
        --gps-start-time ${gps_start} --gps-end-time ${gps_end} \
        --output ${ifo}-INSPIRAL_FULL_DATA_${gps_start}-${seg_duration}.hdf \
        --bank-file ${bank} \
        --ambi-veto-bank-file ${veto_bank} --ambi-status --ambi-snr-threshold 3.0 \
        --ambi-min-filters 10 --ambi-max-filters 20 \
        --frame-files ${site}-${ifo}_GWOSC_4KHZ_R1-1126257415-4096.gwf \
        --verbose
    break
done
        # --trig-start-time 1246067681 --trig-end-time 1246072538 \
        # --filter-inj-only  --injection-window 4.5 \
