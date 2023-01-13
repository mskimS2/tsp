import matplotlib.pyplot as plt


def draw_tsp(pointset, opt_sol):
    """
    optimal tsp solution draw plot
    - axes[0] : tsp pointset (x : 0 ~ 1.0, y : 0 ~ 1.0)
    - axes[1] : tsp solution plot
    """
    print(f"pointset : {pointset}")
    print(f"optimal_solution : {opt_sol}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Pointset fig
    for x, y in pointset:
        axes[0].scatter(x, y, color='lightskyblue', s=20)
        axes[1].scatter(x, y, color='lightskyblue', s=20, zorder=1)
    axes[0].set_title('TSP Pointset', fontsize=25)

    # TSP Solution fig
    for i in range(0, len(opt_sol)-1):
        idx1, idx2 = opt_sol[i], opt_sol[i+1]
        connect_point1 = [pointset[idx1][0], pointset[idx2][0]]
        connect_point2 = [pointset[idx1][1], pointset[idx2][1]]
        axes[1].plot(connect_point1,
                     connect_point2,
                     color='black',
                     zorder=0)

    last = opt_sol[len(opt_sol)-1]
    last_connect_point1 = [pointset[last][0], pointset[0][0]]
    last_connect_point2 = [pointset[last][1], pointset[0][1]]
    axes[1].plot(last_connect_point1,
                 last_connect_point2,
                 color='black',
                 zorder=0)
    axes[1].set_title('TSP Solution', fontsize=25)

    plt.savefig('assets/tsp.PNG')
    plt.show()
