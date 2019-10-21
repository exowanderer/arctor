plt.clf()
plt.scatter(wasp43.trace_xcenters[wasp43.idx_fwd],
            wasp43.trace_ycenters[wasp43.idx_fwd],
            color='C0', label='Forward Scans')
plt.scatter(wasp43.trace_xcenters[wasp43.idx_rev],
            wasp43.trace_ycenters[wasp43.idx_rev],
            color='C1', label='Reverse Scans')

plt.title(
    'Center Positions of the Trace in Forward and Reverse Scanning',
    fontsize=20)
plt.xlabel('X-Center [pixels]', fontsize=20)
plt.ylabel('Y-Center [pixels]', fontsize=20)
