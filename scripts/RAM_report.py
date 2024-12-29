import matplotlib.pyplot as plt

with open('memory_usage.txt', 'r') as f:
    memory_usage = [int(line.strip()) / 2**20 for line in f] 


timestamps = [i / 60 for i in range(len(memory_usage))] 

plt.plot(timestamps, memory_usage)
plt.xlabel('Time, min')
plt.ylabel('Memory Usage (GB)')
plt.title('RAM Usage Over Time')
plt.savefig('memory_usage.png')
plt.close()