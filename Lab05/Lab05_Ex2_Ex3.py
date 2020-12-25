import cherrypy 
import json

class SimpleCalculatorWebService(object):
    exposed = True

    def GET (self, *uri, **query):
        
        if len(uri) != 1:
            raise cherrypy.HTTPError(400, "Provide one command among: 'add', 'sub', 'mul' and 'div'")
        
        if len(query) < 2:
            raise cherrypy.HTTPError(400, "Provide at least 2 operands")

        else:

            operation = uri[0]
            
            operands = []

            for i in range(len(query)):
                operand = query.get('op{}'.format(i + 1))
                if operand==None:
                    raise cherrypy.HTTPError(400, "operand{} is None".format(i+1))
                else:
                    operands.append(float(operand))


            if operation == "add":
                result = 0
                for op in operands:
                    result += op

            elif operation == "sub":
                result = operands[0]
                for i in range(1, len(operands)):
                    result -= operands[i]

            elif operation == "mul":
                result = 1
                for op in operands:
                    result *= op

            elif operation == "div":
                result = operands[0]
                for i in range(1, len(operands)):
                    if operands[i] == 0:
                        raise cherrypy.HTTPError(400, "Operand{} is 0".format(i+1))
                    result /= operands[i]

            else:
                raise cherrypy.HTTPError(501, f"The command {uri} is not supportted")

            output = {
                'command':operation,
                'operands':operands,
                "result":result
            }

            output_json = json.dumps(output)

            return output_json




if __name__ == '__main__':
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			'tools.sessions.on': True
		}
	}
	cherrypy.tree.mount(SimpleCalculatorWebService(), '/', conf)

	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()