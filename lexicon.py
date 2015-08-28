#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import struct
import sys


class Node:
	
	def __init__(self, inf_index=None):
		self.inf_index = inf_index
		self.transitions = None
	
	def add_transition(self, transition_char, next_state_position):
		if self.transitions == None:
			self.transitions = {}
		self.transitions[transition_char] = next_state_position
	
	def get_transition(self, transition_char):
		try:
			return self.transitions[transition_char]
		except KeyError:
			return None

import re

class CompressedForm:
	
	UNIT_REGEX = re.compile(r'([^\s-]+|[\s]|[-])')
	COMPRESSED_UNIT_REGEX = re.compile(r'(?P<del_char_count>[0-9]*)(?P<append_string>[a-zA-Z\\\d]*)')
	UNIT_UNESCAPE_REGEX=re.compile(r'\\(\d)')
	
	def __init__(self, form):
		self.diff_unit_size, self.unit_processors, self.info = CompressedForm.parse_single_compressed_form(form)
		#parse_forms(form.split(','))
		
	def get_lemma(self, word):
		if self.diff_unit_size:
			return self.unit_processors[0](word)
		else:
			units = CompressedForm.UNIT_REGEX.findall(word)
			if len(units) != len(self.unit_processors):
				print("WARNING: len(units) != len(self.unit_processors)")
				return word
			for i, unit in enumerate(units):
				units[i] = self.unit_processors[i](units[i])
			return ("".join(units), self.info)
	
	def parse_forms(forms):
		for form in forms:
			parse_single_compressed_form(form)
		
	def parse_single_compressed_form(form):
		form = form.split('.')
		info = form[1]
		form = form[0]
		
		pos = 0
		if len(form) == 0:
			return False, [lambda u : u], info
		elif form[pos] == '_':
			diff_unit_size = True
			pos += 1
		else:
			diff_unit_size = False
		
		compressed_units = CompressedForm.UNIT_REGEX.findall(form[pos:])
		unit_processors = []
		for c in compressed_units:
			unit_processors += [CompressedForm.parse_compressed_unit(c)]
		return diff_unit_size, unit_processors, info
	
	def parse_compressed_unit(unit):
		if unit in ('-', ' '):
			return lambda u : u
		else:
			l = 0
			m = CompressedForm.COMPRESSED_UNIT_REGEX.search(unit)
			del_char_count = m.group('del_char_count')
			del_char_count =  int(del_char_count) if len(del_char_count) > 0 else 0
			append_string = CompressedForm.UNIT_UNESCAPE_REGEX.sub(r'\1', m.group('append_string'))
			return lambda u : u[0:-del_char_count]+append_string if del_char_count > 0 else u+append_string
				
				

class DELA:
	
	def __init__(self, bin_file, inf_file):
		self.nodes = {}
		print("Loading bin file..")
		self.load_bin_file(bin_file)
		print("Loading inf file..")
		self.load_inf_file(inf_file)
		
	def load_inf_file(self, bin_file):
		with open(bin_file, encoding='utf-16-le') as f:
			inf = [line.rstrip('\n') for line in f]
		header = int(inf[0].lstrip('\ufeff'))
		if header != len(inf)-1:
			print("Invalid header: %d Actual: %d" % (header, len(inf)-1))
		self.inf = inf[1:]
		print("INF LOADED")
	
	def load_bin_file(self, bin_file):
		with open(bin_file, "rb") as f:
			file_size = struct.unpack('>i', f.read(4))[0]
			print("Size: %d bytes" % file_size)
			while f.tell() < file_size:
				self.load_automaton_state(f)
#		print(self.nodes[4].get_transition('r'))
#		print(self.nodes[679750].get_transition('o'))
#		print(self.nodes[712152].get_transition('u'))
#		print(self.nodes[715670].get_transition('b'))
#		print(self.nodes[715717].get_transition('o'))
#		print(self.nodes[5856].get_transition('u'))
#		print(self.nodes[1242].inf_index)
#		print(self.find('roubou'))
		
	
	def find(self, word):
		pos = 4
		for c in word:
			pos = self.nodes[pos].get_transition(c)
		inf_index = self.nodes[pos].inf_index
		if inf_index == None:
			raise KeyError('"%s" not found' % word)
		inf_value = self.inf[inf_index]
		if type(inf_value) == str:
			cunits = inf_value.split(",")
			inf_value = []
			self.inf[inf_index] = inf_value
			for cu in cunits:
				inf_value += [CompressedForm(cu)]
		return inf_value
	
	def get_lemmas(self, word, noEcho=False):
		try:
			cfs = self.find(word)
			return [cf.get_lemma(word) for cf in cfs]
		except KeyError:
			return None if noEcho else [(word, None)]
			
	def load_automaton_state(self, f):
		state_position = f.tell()
		state_header = struct.unpack('>H', f.read(2))[0]
		#print("State Header: %d" % state_header)
		is_final = 0b1000000000000000 & state_header != 0b1000000000000000
		num_transitions = ~0b1000000000000000 & state_header
		#print("Number of transitions: %d" % num_transitions)
		if is_final:
			inf_index = struct.unpack('>i', b'\0' + f.read(3))[0]
			#print("Inf Index: %d" % inf_index)
			node = Node(inf_index)
		else:
			node = Node()
		for i in range(num_transitions):
			transition_char, next_state_position = self.load_transition(f)
			node.add_transition(transition_char, next_state_position)
		self.nodes[state_position] = node
	
	def load_transition(self, f):
		transition_char = f.read(2).decode('utf-16-be')
		#print(transition_char)
		next_state_position = struct.unpack('>i', b'\0' + f.read(3))[0]
		#print("Next State Position: %d" % next_state_position)
		return transition_char, next_state_position
			
			

import os
class Lexicon:
	
	def __init__(self, source):
		if type(source) == list:
			pass
		else:
			source = Lexicon.find_delas(source)
		self.delas = []
		for bin_file, inf_file in source:
			dela = DELA(bin_file, inf_file)
			self.delas += [dela]
			print("LOADED DELA")
	
	def get_lemmas(self, word):
		ret = []
		for dela in self.delas:
			lemmas = dela.get_lemmas(word)
			if len(lemmas) > 0 and lemmas[0][1] != None:
				ret.extend(lemmas)
		return set(ret)
	
	def find_delas(path):
		files = []
		for (dirpath, dirnames, filenames) in os.walk(path):
			files.extend(filenames)
			break
		files = [os.path.join(path, f) for f in files]
		candidates = {}
		for f in files:
			if f.lower()[-4:] == '.bin':
				candidates[f.lower()[:-4]] = (f,)
		for f in files:
			key = f.lower()[:-4]
			if f.lower()[-4:] == '.inf' and key in candidates:
				candidates[key] += (f,)
		return [f for f in candidates.values() if len(f) == 2]