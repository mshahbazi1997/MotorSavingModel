import os
import datetime
import matplotlib.pyplot as plt
import motornet as mn


def create_directory(directory_name=None):
    if directory_name is None:
        directory_name = datetime.datetime.now().date().isoformat()

    # Get the user's home directory
    home_directory = os.path.expanduser("~")

    # Create the full directory path
    directory_path = os.path.join(home_directory, "Documents", "Data", directory_name)

    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    # Return the created directory's name (whether it was newly created or already existed)
    return directory_path


def plot_training_log(log):
  ax = plt.subplot(1,1,1)

  ax.semilogy(log)

  ax.set_ylabel("Loss")
  ax.set_xlabel("Batch #")
  return ax


def plot_simulations(xy, target_xy):
  target_x = target_xy[:, -1, 0]
  target_y = target_xy[:, -1, 1]

  plt.figure(figsize=(5,3))

  plt.subplot(1,1,1)
  plt.ylim([0.3, 0.65])
  plt.xlim([-0.3, 0.])

  plotor = mn.plotor.plot_pos_over_time


  plotor(axis=plt.gca(), cart_results=xy)
  plt.scatter(target_x, target_y)

  #plt.subplot(1,2,2)
  #plt.ylim([-0.1, 0.1])
  #plt.xlim([-0.1, 0.1])
  #plotor(axis=plt.gca(), cart_results=xy - target_xy)
  #plt.axhline(0, c="grey")
  #plt.axvline(0, c="grey")
  #plt.xlabel("X distance to target")
  #plt.ylabel("Y distance to target")
  plt.show()




        