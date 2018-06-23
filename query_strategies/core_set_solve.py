import pickle
from datetime import datetime
from gurobipy import *
import pdb

def solve_fac_loc(xx, yy, subset, n, budget):
    t_start = datetime.now()
    model = Model("k-center")
    x={}
    y={}
    z={}

    print 'gen z', datetime.now()-t_start
    for i in range(n):
        # z_i: is a loss
        z[i] = model.addVar(obj=1, ub=0.0, vtype="B", name="z_{}".format(i))
     
    print 'gen x y', datetime.now()-t_start
    m = len(xx)
    for i in range(m):
        if i % 1000000 == 0:
            print 'gen x y {}/{}'.format(i, m), datetime.now()-t_start
        _x = xx[i]
        _y = yy[i]
        # y_i = 1 means i is facility, 0 means it is not
        if _y not in y:
            if _y in subset: 
                y[_y] = model.addVar(obj=0, ub=1.0, lb=1.0, vtype="B", name="y_{}".format(_y))
            else:
                y[_y] = model.addVar(obj=0, vtype="B", name="y_{}".format(_y))
        #if not _x == _y:
        x[_x,_y] = model.addVar(obj=0, vtype="B", name="x_{},{}".format(_x,_y))
    model.update()

    print 'gen sum q', datetime.now()-t_start
    coef = [1 for j in range(n)]
    var = [y[j] for j in range(n)]
    model.addConstr(LinExpr(coef,var), "=", rhs=budget+len(subset), name="k_center")

    print 'gen <=', datetime.now()-t_start
    for i in range(m):
        if i % 1000000 == 0:
            print 'gen <= {}/{}'.format(i, m), datetime.now()-t_start
        _x = xx[i]
        _y = yy[i]
        #if not _x == _y:
        model.addConstr(x[_x,_y], "<=", y[_y], name="Strong_{},{}".format(_x,_y))

    print 'gen sum 1', datetime.now()-t_start
    yyy = {}
    for v in range(m):
        if i % 1000000 == 0:
            print 'gen sum 1 {}/{}'.format(i, m), datetime.now()-t_start
        _x = xx[v]
        _y = yy[v]
        if _x not in yyy:
            yyy[_x]=[]
        if _y not in yyy[_x]:
            yyy[_x].append(_y)

    for _x in yyy:
        coef = []
        var = []
        for _y in yyy[_x]:
            #if not _x==_y:
            coef.append(1)
            var.append(x[_x,_y])
        coef.append(1)
        var.append(z[_x])
        model.addConstr(LinExpr(coef,var), "=", 1, name="Assign{}".format(_x))

    print 'ok', datetime.now()-t_start
    model.__data = x,y,z
    return model

r_name = 'mip.pkl'
w_name = 'sols.pkl'
print 'load pickle {}'.format(r_name)
xx, yy, dd, subset, max_dist, budget, n = pickle.load(open(r_name, 'rb'))
print len(subset), budget, n


t_start = datetime.now()

print 'start'
ub = max_dist
lb = ub/2.0

model = solve_fac_loc(xx, yy, subset, n, budget)


#model.setParam( 'OutputFlag', False )
x,y,z = model.__data
tor = 1e-7
sols = None
while ub-lb > tor:
    cur_r = (ub+lb)/2.0
    print "======[State]======", ub, lb, cur_r, ub-lb

    # viol = numpy.where(_d>cur_r)
    viol = [i for i in range(len(dd)) if dd[i]>cur_r]
    # new_max_d = numpy.min(_d[_d>=cur_r])
    new_max_d = min([d for d in dd if d >= cur_r])
    # new_min_d = numpy.max(_d[_d<=cur_r])
    new_min_d = max([d for d in dd if d <= cur_r])
    print "If it succeeds, new max is:", new_max_d, new_min_d
    for v in viol:
        x[xx[v], yy[v]].UB = 0

    model.update()
    r = model.optimize()
    if model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
        failed = True
        print "======[Infeasible]======"
    elif sum([z[i].X for i in range(len(z))]) > 0:
        failed = True
        print "======[Failed]======Failed"
    else:
        failed = False
    if failed:
        lb = max(cur_r, new_max_d)
        #failed so put edges back
        for v in viol:
            x[xx[v], yy[v]].UB = 1
    else:
        print "======[Solution Founded]======", ub, lb, cur_r
        ub = min(cur_r, new_min_d)
        sols = [v.varName for v in model.getVars() if v.varName.startswith('y') and v.x>0]

print 'end', datetime.now()-t_start, ub, lb, max_dist

if sols is not None:
    sols = [int(v.split('_')[-1]) for v in sols]
print 'save pickle {}'.format(w_name)
pickle.dump(sols, open(w_name, 'wb'), 2)

