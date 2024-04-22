awk -F',' '{ops += (64*($(NF-1)+(2*$(NF-2))+$(NF-3)+$(NF-4)) + 512*$NF)} END {print ops}' $@
