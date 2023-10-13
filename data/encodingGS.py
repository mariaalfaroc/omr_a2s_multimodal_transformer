import re
import numpy as np


# TODO:
# - Comprobar que funciona con GrandStaff (todas las opciones de codificación)


ENCODING_OPTIONS = ['kern', 'bekern']


class krnConverter():
    """Main Kern converter operations class."""
    
    def __init__(self, encoding: str = 'bekern', keep_ligatures: bool = True):
        self.header_word = '**kern' if encoding == 'kern' else '**bekern'
        self.reserved_words = ['clef', 'k[', '*M']
        self.reserved_dot = '.'
        self.reserved_dot_EncodedCharacter = 'DOT'
        self.clef_change_other_voices = '*'
        self.open_spine = '*^'
        self.close_spine = '*v'
        self.comment_symbols = ['*', '!']
        self.voice_change = '\t'
        self.step_change = '\n'
        self.keep_ligatures = keep_ligatures

        # Convert function
        assert encoding in ENCODING_OPTIONS, f'You must chose one of the possible encoding options: {",".join(ENCODING_OPTIONS)}'
        self.encoding = encoding
        self.convert_step = self.cleanKernFile


    def _readSrcFile(self, file_path: str) -> list:
        """Read polyphonic kern file and adequate the format for further processes."""
        with open(file_path) as fin:
            in_src = fin.read().splitlines()
        
        return np.array(in_src)


    def _postprocessKernSequence(self, in_score: list) -> list:
        """Exchanging '*' for the actual symbol; removing ligatures (if so)"""

        # Retrieving positions with '*':
        positions_elements = list(np.where(np.array([('*' in line) and ('*^' not in line) and ('*v' not in line) for line in in_score]) == True)[0])
        positions_voices = [np.where(np.array(in_score[idx]) == '*')[0] for idx in positions_elements]

        # For each position, we retrieve the last explicit clef symbol and include it in the stream:
        for it_position in range(len(positions_elements)):
            position_element = positions_elements[it_position]
            for position_voice in positions_voices[it_position]:
                

                # Locating last position with the required number of voices (in case spines open/close):
                it_reference = position_element
                end_search = False
                while(it_reference >= 0 and end_search == False):
                    if len(in_score[it_reference]) >= position_voice + 1:
                        it_reference -= 1
                    else:
                        it_reference += 1
                        end_search = True
                    pass
                pass


                previous_elements = [line[position_voice] for line in in_score[it_reference:position_element]]
                try:
                    new_element = in_score[it_reference + max(np.where(np.char.startswith(previous_elements, '*clef')))[0]][position_voice]
                except:
                    new_element = in_score[position_element][position_voice-1]
                in_score[position_element][position_voice] = new_element
            pass
        pass


        # Removing ligatures (if so):
        ### Locating openings:
        if not self.keep_ligatures:
            for it_voice in range(len(in_score)):
                for pos in np.where(np.char.startswith([line[it_voice] for line in in_score], '['))[0]:
                    in_score[pos][it_voice] = in_score[pos][it_voice].replace('[', '')
                pass

                for pos in np.where(np.char.endswith([line[it_voice] for line in in_score], ']'))[0]:
                    if not np.char.startswith(in_score[pos][it_voice], '*'): 
                        in_score[pos][it_voice] = in_score[pos][it_voice].replace(']', '')
                    pass
                pass
            pass
        pass

        return in_score


    def cleanKernFile(self, file_path: str) -> list:
        """Convert complete kern sequence to CLEAN kern format."""
        in_file = self._readSrcFile(file_path = file_path)

        # Processing individual voices:
        out_score = list()
        for step in in_file:
            # Splitting voices:
            voices = step.split("\t")
            
            # Iterating through voices:
            current_step = list()
            for single_voice in voices:
                try:
                    current_step.append(" ".join([self.cleanKernToken(u) for u in single_voice.split(" ")]))
                except:
                    pass
            if len(current_step) > 0: out_score.append(current_step)
        pass

        # Postprocess obtained score:
        out_score = self._postprocessKernSequence(out_score)

        return out_score


    def cleanKernToken(self, in_token: str) -> str:

        """Convert a kern token to its CLEAN equivalent."""
        out_token = None                                                    # Default

        if any([u in in_token for u in self.reserved_words]):               # Relevant reserved tokens
            out_token = in_token
        
        elif in_token == self.reserved_dot:                                 # Case when using '.' for sync. voices
            out_token = self.reserved_dot_EncodedCharacter

        elif in_token.strip() == self.clef_change_other_voices:             # Clef change in other voices
            out_token = in_token

        elif in_token.strip() == self.open_spine or in_token.strip() == self.close_spine: # Open/close spine
            out_token = in_token

        elif any([in_token.startswith(u) for u in self.comment_symbols]):   # Comments
            out_token = None

        elif in_token.startswith('s'):                                      # Slurs
            out_token = 's'

        elif '=' in in_token:                                               # Bar lines
            out_token = '='            

        elif not 'q' in in_token:
            if 'rr' in in_token:                     
                in_token = in_token.replace("·", "")                        # Multirest
                out_token = re.findall('rr[0-9]+', in_token)[0]
            elif 'r' in in_token:    
                in_token = in_token.replace("·", "")                        # Rest
                out_token = in_token.split('r')[0]+'r'
            else:                     
                in_token = in_token.replace("·", "")                        # Music note
                out_token = re.findall('\[*\d+[.]*[a-gA-G]+[n#-]*\]*', in_token)[0]
        
        elif 'q' in in_token:
            in_token = in_token.replace("·", "")                            # Music note with q
            out_token = re.findall('\[*\d*[a-gA-G]+[n#-]*[q]+\]*', in_token)[0]
        return out_token
    
    
    # ---------------------------------------------------------------------------- CONVERT CALL
    
    def convert(self, src_file: str ) -> list:
        out = self.convert_step(src_file).T
        out_line = self.step_change.join([self.voice_change.join(voice) for voice in out])
        return out_line



if __name__ == '__main__':
    import os
    conv = krnConverter()

    ### Checking all files:
    # base_path = 'data/'
    # with open('dst.txt', 'w') as fout:
    #     for root, dir_names, file_names in os.walk(base_path):
    #         for single_file in file_names:
    #             if single_file.endswith('bekrn'):
    #                 target_file = os.path.join(root, single_file)
    #                 try:
    #                     res = conv.cleanKernFile(target_file)
    #                     fout.write("{} - {}\n".format(target_file, "Done!"))
    #                 except:
    #                     fout.write("{} - {}\n".format(target_file, "Fail!"))

    # path = 'data/grandstaff/chopin/mazurkas/mazurka33-3/maj3_down_m-33-36.bekrn'

    path = 'data/grandstaff/chopin/mazurkas/mazurka33-3/maj2_up_m-15-18.bekrn'
    res = conv.cleanKernFile(path)
    print("end")