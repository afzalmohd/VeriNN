from z3 import *
value = 0
set_sz = 0
def change_to_sat_format(s,eta_set, f, mode, layer, epsilon, layrewise_size, ls_obj, img, nodes):
        ls_i = 0 
        lbound = []
        ubound = []
        ind  = 0
        global value
        global set_sz
        if mode == 2:
                print("nodes where new etas are created")
        for line in f:
                if line.find(':=') != -1:
                        y = line.split(':=')
                        node = (y[0].strip())
                        set_sz = len(eta_set)
                        if mode == 2:
                                node = node + '_' + str(layer)
                        node = Real(node)
                        nodes.append(node)
                        con = (y[1].strip())
                        newcon = con.split('+')
                        i = 0
                        expr = 0
                        while i < len(newcon):
                                newconn = newcon[i].strip()
                                if newconn.find("eps") == -1:
                                        newconn = round(float(newconn),4)
                                        expr = expr + RealVal(newconn)
                                else:
                                        coef = newconn.split('.(')[0]
                                        coef = round(float(coef),4)
                                        varr = newconn.split('.(')[1]
                                        var = varr.replace(')', '')
                                        numb = var[3:]
                                        numb = int(numb)
                                        '''first time for outputs'''
                                        if mode == 1:
                                                
                                                eta_set.add(numb)
                                                var = Real(var)
                                                if numb >= 0 and numb < 784:
                                                        s.add(epsilon*var + img[numb] >= 0,epsilon*var + img[numb] <= 1)
                                                s.add(var <= 1 , -1<=var)
                                        elif mode == 2: 
                                                eta_set.add(numb)
                                                value = var
                                                var = var + 'dd'        
                                                var = Real(var)
                                                if numb == 787 or numb == 868:
                                                        s.add(1 < var, -1 > var)

                                        else:
                                                
                                                var = var + 'dd'
                                                var = Real(var)

                                        
                                        expr = expr + (RealVal(coef) * var)
                                i = i + 1
                        s.add(node == expr) 
                else:
                        lb = RealVal( line.split(',')[0] )
                        ubb =  line.split(',')[1]
                        ub = RealVal( ubb.strip('\n') )
                        lbound.append(lb)
                        ubound.append(ub)
                        if mode == 1:
                                s.add(nodes[ind] <= ub, nodes[ind] >= lb)
                        elif mode == 2:
                                # print(nodes[ind] == ls_obj[layer][ls_i])
                                # print(str(set_sz) + '   ' + str(len(eta_set)))
                                # print(value)

                                if set_sz < len(eta_set):
                                        s.add(nodes[ind] == ls_obj[layer][ls_i])
                                        print(str(value) + " in layer " + str(layer + 1) + " at node " + str(ls_i))  
                                ls_i = ls_i + 1

                                if ls_i == layrewise_size[layer]:
                                        ls_i=0
                                        layer+=1
                        else:
                                #s.add(nodes[ind] == ls_obj[layer][ls_i])
                                s.add(nodes[ind] <= ub, nodes[ind] >= lb)
                                ls_i = ls_i + 1
                        ind = ind + 1

def add_maxsat_cons(dbg_solv, s, eta_set, eta_dd,f, mode, layer,layrewise_size,nodes ):
        ls_i = 0 
        lbound = []
        ubound = []
        ind  = 0
        global value 
        global set_sz
        value = 0
        for line in f:
                if line.find(':=') != -1:
                        y = line.split(':=')
                        node = (y[0].strip())
                        set_sz = len(eta_set)
                        if mode == 2:
                                node = node + '_' + str(layer)
                        node = Real(node)
                        nodes.append(node)
                        con = (y[1].strip())
                        newcon = con.split('+')
                        i = 0
                        expr = 0
                        while i < len(newcon):
                                newconn = newcon[i].strip()
                                if newconn.find("eps") == -1:
                                        newconn = round(float(newconn),4)
                                        expr = expr + RealVal(newconn)
                                else:
                                        coef = newconn.split('.(')[0]
                                        coef = round(float(coef),4)
                                        varr = newconn.split('.(')[1]
                                        var = varr.replace(')', '')
                                        numb = var[3:]
                                        numb = int(numb)
                                        if mode == 2:
                                                
                                                eta_set.add(numb)
                                                value = var   
                                                var = Real(var)

                                        else:
                                                var = Real(var)

                                        
                                        expr = expr + (RealVal(coef) * var)
                                i = i + 1
                        
                        s.add(node == expr) 
                        
                        dbg_solv.add(node == expr)

                else:
                        lb = RealVal( line.split(',')[0] )
                        ubb =  line.split(',')[1]
                        ub = RealVal( ubb.strip('\n') )
                        lbound.append(lb)
                        ubound.append(ub)
                        if mode == 2:
                                
                                if set_sz < len(eta_set):
                                        
                                        t = str(nodes[ind]) + '_b'
                                        t = Bool(t)
                                        newvalue = str(value) + 'dd'
                                        
                                        value = Real(value)
                                        
                                        s.add_soft(t)
                                        s.add(Implies(t, (value == eta_dd[newvalue])))
                                        
                                ls_i = ls_i + 1
                                if ls_i == layrewise_size[layer]:
                                        ls_i=0
                                        layer+=1
                        else:
                                #s.add(nodes[ind] == ls_obj[layer][ls_i])
                                s.add(nodes[ind] <= ub, nodes[ind] >= lb)
                                ls_i = ls_i + 1
                        ind = ind + 1
